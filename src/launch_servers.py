#!/usr/bin/env python3
"""Launch one vLLM OpenAI server per GPU and write an endpoint manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-model", type=Path, default=Path("/data/project/sustech/model/Qwen2.5-7B-Instruct"))
    parser.add_argument("--weak-model", type=Path, default=Path("/data/project/sustech/model/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "server_endpoints.json")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "server_logs")
    parser.add_argument("--strong-replicas", type=int, default=5)
    parser.add_argument("--strong-name", default="qwen25-7b")
    parser.add_argument("--weak-name", default="qwen25-1_5b")
    parser.add_argument("--skip-weak", action="store_true")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index in range(args.strong_replicas):
        jobs.append(("strong", index, index, args.base_port + index, args.strong_model, args.strong_name, "0.88"))
    if not args.skip_weak:
        jobs.append(("weak", args.strong_replicas, args.strong_replicas, args.base_port + args.strong_replicas, args.weak_model, args.weak_name, "0.85"))
    processes = []
    for role, index, gpu, port, model, served_name, memory in jobs:
        log_path = args.log_dir / f"{role}{index}.log"
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0", "FLASHINFER_DISABLE_VERSION_CHECK": "1"})
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
                   "--host", "127.0.0.1", "--port", str(port), "--served-model-name", served_name,
                   "--dtype", "bfloat16", "--attention-backend", "TRITON_ATTN", "--enforce-eager", "--max-model-len", "8192", "--gpu-memory-utilization", memory,
                   "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", str(args.max_num_batched_tokens), "--trust-remote-code"]
        log = log_path.open("w")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append({"role": role, "gpu": gpu, "port": port, "pid": process.pid, "model": str(model),
                          "served_model": served_name, "log": str(log_path), "process": process, "log_handle": log})
        print(f"Started {role}{index} pid={process.pid} gpu={gpu} port={port}", flush=True)
    deadline = time.time() + args.wait_seconds
    ready = set()
    while len(ready) < len(processes) and time.time() < deadline:
        for item in processes:
            key = item["port"]
            if key in ready:
                continue
            if item["process"].poll() is not None:
                for process in processes:
                    process["log_handle"].close()
                raise RuntimeError(f"Server exited unexpectedly; inspect {item['log']}")
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{key}/v1", timeout=5) as client:
                    if client.get("/models").is_success:
                        ready.add(key)
                        print(f"Ready {item['role']} port={key}", flush=True)
            except httpx.HTTPError:
                pass
        if len(ready) < len(processes):
            time.sleep(5)
    for item in processes:
        item["log_handle"].close()
        item.pop("process")
        item.pop("log_handle")
    if len(ready) != len(processes):
        raise TimeoutError(f"Only {len(ready)}/{len(processes)} servers became ready")
    payload = {"strong_endpoints": [f"http://127.0.0.1:{x['port']}/v1" for x in processes if x["role"] == "strong"],
               "weak_endpoints": [f"http://127.0.0.1:{x['port']}/v1" for x in processes if x["role"] == "weak"],
               "strong_model": args.strong_name, "weak_model": args.weak_name, "processes": processes}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
