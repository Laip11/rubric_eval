"""Multi-process offline vLLM inference with whole-stage batch submission."""
from __future__ import annotations

import asyncio
import gc
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

from async_vllm import LLMResult


def _serialize_output(output: Any) -> dict[str, Any]:
    completion = output.outputs[0]
    logprob_groups = completion.logprobs
    entries = []
    if logprob_groups is not None:
        for index, token_id in enumerate(completion.token_ids):
            candidates = logprob_groups[index] if index < len(logprob_groups) else {}
            own = candidates.get(token_id)
            token = getattr(own, "decoded_token", None) if own is not None else None
            top = []
            for candidate in candidates.values():
                decoded = getattr(candidate, "decoded_token", None)
                if decoded is not None:
                    top.append({"token": decoded, "logprob": float(candidate.logprob)})
            entries.append({"token": token or "", "logprob": float(own.logprob) if own is not None else float("-inf"),
                            "top_logprobs": top})
    return {"text": completion.text, "logprobs": entries,
            "prompt_tokens": len(output.prompt_token_ids), "completion_tokens": len(completion.token_ids),
            "finish_reason": getattr(completion, "finish_reason", None)}


def _worker(role: str, config: dict[str, Any], requests_queue: Any, results_queue: Any, ready_queue: Any) -> None:
    try:
        os.setsid()
        os.environ.update({"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in config["gpus"]),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                           "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                           "VLLM_USE_FLASHINFER_SAMPLER": "0", "FLASHINFER_DISABLE_VERSION_CHECK": "1"})
        from vllm import LLM
        from vllm.sampling_params import StructuredOutputsParams

        def load_engine(engine_config: dict[str, Any]) -> tuple[Any, Any]:
            engine = LLM(model=str(engine_config["model_path"]), tensor_parallel_size=len(engine_config["gpus"]),
                         dtype="bfloat16", attention_backend="TRITON_ATTN", enforce_eager=True,
                         max_model_len=int(engine_config.get("max_model_len", 16384)),
                         gpu_memory_utilization=engine_config["memory"], max_num_seqs=engine_config["max_num_seqs"],
                         max_num_batched_tokens=max(65536, int(engine_config.get("max_model_len", 16384))),
                         trust_remote_code=True)
            return engine, engine.get_default_sampling_params()

        dynamic_roles = config.get("dynamic_roles")
        current_role = role
        loaded_role = None
        llm = None
        default_sampling = None
        if dynamic_roles is None:
            llm, default_sampling = load_engine(config)
        ready_queue.put({"status": "ready", "role": role})
        while True:
            job = requests_queue.get()
            if job is None:
                break
            if dynamic_roles is None:
                job_id, requests = job
            else:
                job_id, current_role, requests = job
            started = time.perf_counter()
            try:
                if dynamic_roles is not None and (llm is None or current_role != getattr(llm, "_paor_role", None)):
                    if llm is not None:
                        shutdown = getattr(llm, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                        del llm
                        gc.collect()
                        import torch
                        torch.cuda.empty_cache()
                    engine_config = {**dynamic_roles[current_role], "gpus": config["gpus"]}
                    llm, default_sampling = load_engine(engine_config)
                    llm._paor_role = current_role
                messages = [[{"role": "system", "content": request["system"]},
                             {"role": "user", "content": request["user"]}] for request in requests]
                sampling_params = []
                for request in requests:
                    params: Any = default_sampling.clone()
                    params.temperature = float(request.get("temperature", 0.0))
                    params.max_tokens = int(request.get("max_tokens", 512))
                    params.seed = int(request["seed"]) if request.get("seed") is not None else None
                    params.logprobs = 5 if request.get("logprobs") else None
                    if request.get("structured_json") is not None:
                        params.structured_outputs = StructuredOutputsParams(
                            json=request["structured_json"], disable_additional_properties=True)
                    sampling_params.append(params)
                outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=True,
                                   chat_template_kwargs={"enable_thinking": False})
                results = [_serialize_output(output) for output in outputs]
                if len(results) != len(requests):
                    raise RuntimeError(f"Expected {len(requests)} outputs, received {len(results)}")
                results_queue.put({"job_id": job_id, "results": results, "elapsed": time.perf_counter() - started})
            except Exception:
                results_queue.put({"job_id": job_id, "error": traceback.format_exc()})
    except Exception:
        ready_queue.put({"status": "error", "role": role, "error": traceback.format_exc()})


class OfflineLLMPool:
    def __init__(self, role_configs: dict[str, dict[str, Any]], *, wait_seconds: int = 1800,
                 dynamic_gpus: list[int] | None = None) -> None:
        self.stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0})
        self._context = mp.get_context("spawn")
        self._workers: dict[str, list[dict[str, Any]]] = {role: [] for role in role_configs}
        self._job_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._job_id = 0
        self._dynamic_workers: list[dict[str, Any]] = []
        self._switch_lock = threading.Lock()
        try:
            if dynamic_gpus is not None:
                self._workers = {"dynamic": self._dynamic_workers}
                for gpu in dynamic_gpus:
                    config = {"gpus": [gpu], "dynamic_roles": role_configs}
                    requests_queue = self._context.Queue()
                    results_queue = self._context.Queue()
                    ready_queue = self._context.Queue()
                    process = self._context.Process(target=_worker, args=("dynamic", config, requests_queue, results_queue, ready_queue))
                    process.start()
                    self._dynamic_workers.append({"process": process, "requests": requests_queue, "results": results_queue,
                                                  "ready": ready_queue, "config": config, "replica": gpu,
                                                  "lock": threading.Lock(), "dynamic": True})
                for worker in self._dynamic_workers:
                    try:
                        message = worker["ready"].get(timeout=wait_seconds)
                    except queue.Empty as exc:
                        raise TimeoutError(f"Dynamic GPU worker did not become ready within {wait_seconds}s") from exc
                    if message.get("status") != "ready":
                        raise RuntimeError(f"Dynamic GPU worker failed:\n{message.get('error', 'unknown error')}")
                    print(f"Dynamic GPU worker ready on GPU {worker['config']['gpus']}", flush=True)
                return
            for role, config in role_configs.items():
                gpu_sets = ([[gpu] for gpu in config["gpus"]] if config.get("replica_per_gpu") else [config["gpus"]])
                for replica, gpus in enumerate(gpu_sets):
                    replica_config = {**config, "gpus": gpus}
                    requests_queue = self._context.Queue()
                    results_queue = self._context.Queue()
                    ready_queue = self._context.Queue()
                    process = self._context.Process(target=_worker, args=(role, replica_config, requests_queue, results_queue, ready_queue))
                    process.start()
                    self._workers[role].append({"process": process, "requests": requests_queue, "results": results_queue,
                                                "ready": ready_queue, "config": replica_config, "replica": replica,
                                                "lock": threading.Lock()})
            for role, workers in self._workers.items():
                for worker in workers:
                    try:
                        message = worker["ready"].get(timeout=wait_seconds)
                    except queue.Empty as exc:
                        raise TimeoutError(f"Offline {role} engine did not become ready within {wait_seconds}s") from exc
                    if message.get("status") != "ready":
                        raise RuntimeError(f"Offline {role} engine failed:\n{message.get('error', 'unknown error')}")
                    print(f"Offline {role} replica {worker['replica']} ready on GPU {worker['config']['gpus']}: "
                          f"{worker['config']['model_path']}", flush=True)
        except Exception:
            self._close_sync()
            raise

    @staticmethod
    def request(system: str, user: str, *, role: str = "strong", temperature: float = 0.0,
                max_tokens: int = 512, seed: int | None = None, logprobs: bool = False,
                tag: str = "global", structured_json: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"system": system, "user": user, "role": role, "temperature": temperature,
                "max_tokens": max_tokens, "seed": seed, "logprobs": logprobs, "tag": tag,
                "structured_json": structured_json}

    def _call_worker(self, role: str, worker: dict[str, Any], values: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, LLMResult]]:
        requests = [request for _, request in values]
        with worker["lock"]:
            with self._job_lock:
                self._job_id += 1
                job_id = self._job_id
            worker["requests"].put((job_id, role, requests) if worker.get("dynamic") else (job_id, requests))
            while True:
                response = worker["results"].get()
                if response.get("job_id") == job_id:
                    break
        if "error" in response:
            with self._stats_lock:
                for request in requests:
                    self.stats[request.get("tag", "global")]["errors"] += 1
            raise RuntimeError(f"Offline {role} batch failed:\n{response['error']}")
        elapsed = float(response["elapsed"])
        output = []
        for (index, request), value in zip(values, response["results"]):
            tag = request.get("tag", "global")
            with self._stats_lock:
                self.stats[tag]["calls"] += 1
                self.stats[tag]["prompt_tokens"] += int(value["prompt_tokens"])
                self.stats[tag]["completion_tokens"] += int(value["completion_tokens"])
                self.stats[tag]["seconds"] += elapsed
                self.stats[tag][f"tokens:{role}"] = self.stats[tag].get(f"tokens:{role}", 0) + int(value["prompt_tokens"]) + int(value["completion_tokens"])
                self.stats[tag][f"calls:{role}"] = self.stats[tag].get(f"calls:{role}", 0) + 1
            output.append((index, LLMResult(str(value["text"]), value["logprobs"], int(value["prompt_tokens"]),
                                            int(value["completion_tokens"]), elapsed, f"offline:{role}",
                                            finish_reason=value.get("finish_reason"))))
        return output

    def _call_sync(self, role: str, requests: list[dict[str, Any]]) -> list[LLMResult]:
        dynamic_workers = getattr(self, "_dynamic_workers", [])
        guard = self._switch_lock if dynamic_workers else nullcontext()
        with guard:
            workers = dynamic_workers or self._workers[role]
            indexed = list(enumerate(requests))
            assignments = [indexed[i::len(workers)] for i in range(len(workers))]
            assignments = [values for values in assignments if values]
            if len(assignments) == 1:
                output = self._call_worker(role, workers[0], assignments[0])
            else:
                with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                    futures = [executor.submit(self._call_worker, role, worker, values)
                               for worker, values in zip(workers, assignments)]
                    output = [item for future in futures for item in future.result()]
            return [result for _, result in sorted(output, key=lambda item: item[0])]

    async def complete_many(self, requests: list[dict[str, Any]], *, chunk_size: int | None = None) -> list[LLMResult]:
        if not requests:
            return []
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, request in enumerate(requests):
            grouped[request.get("role", "strong")].append((index, request))

        async def submit(role: str, values: list[tuple[int, dict[str, Any]]]) -> tuple[list[int], list[LLMResult]]:
            indices = [index for index, _ in values]
            role_requests = [request for _, request in values]
            return indices, await asyncio.to_thread(self._call_sync, role, role_requests)

        results: list[LLMResult | None] = [None] * len(requests)
        for indices, values in await asyncio.gather(*(submit(role, values) for role, values in grouped.items())):
            for index, value in zip(indices, values):
                results[index] = value
        return [value for value in results if value is not None]

    def tag_snapshot(self, tag: str) -> dict[str, float]:
        with self._stats_lock:
            return dict(self.stats.get(tag, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0}))

    def total_snapshot(self) -> dict[str, float]:
        keys = ["calls", "prompt_tokens", "completion_tokens", "seconds", "errors"]
        with self._stats_lock:
            return {key: sum(values[key] for values in self.stats.values()) for key in keys}

    def _close_sync(self) -> None:
        for workers in getattr(self, "_workers", {}).values():
            for worker in workers:
                process = worker["process"]
                if process.is_alive():
                    worker["requests"].put(None)
        for workers in getattr(self, "_workers", {}).values():
            for worker in workers:
                process = worker["process"]
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self._workers.clear()

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)
