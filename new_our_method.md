
# 方法规范：基于多视图聚类共识与残差正交筛选的动态准则生成 (MCCR-Eval)

> **当前主方法是第 5 节的 MCCR-ME（最小证据集合）。** 第 1–3 节（v1）和第 4 节（v2）作为前作和消融保留。

> **2026-11-08 修订说明。** 第 1–3 节是 MCCR v1（代码中的 `mccr`）的原始规范，已删去不成立的“严格最小 / 无冗余”断言，改为经验性描述。审查结论见 `docs/MCCR_REVIEW.md`。第 4 节新增 **MCCR v2**（`mccr_v2`）：方向性弱监督筛选、白化加权和可证明的词典序聚合。两个版本共享第 2.1–2.2 节的多视图生成与聚类；评测协议（离散分数、解析失败计错、共享探测）见 `method.md`。

## 1. 核心思想与第一性原理动机

在构建针对用户指令 $q$ 的动态评估准则（Rubrics）时，理想的准则集合应当由两部分构成：**最小证据覆盖集（Minimal Evidence Coverage, $\mathcal{R}_{\text{core}}$）**与**响应触发的判别补充集（Response-Triggered Supplementary Rubrics, $\mathcal{R}_{\text{ext}}$）**。

从第一性原理出发，直接依赖单次 Prompt 让模型生成“最小完备覆盖”存在两个固有缺陷：

1. **单视角盲区与冗余并存**：仅看指令 $q$（零样本先验）容易遗漏只有在具体回答语境下才会显现的隐性必要约束；而引入回答 $(q, y_m)$ 后，模型又会产生大量同义反复或被单一回答带偏的噪声准则。
2. **主干与分支的界限模糊**：哪些准则是所有回答都必须满足的“及格底线”，哪些准则是区分高下或捕获特定缺陷的“额外检验”，无法仅凭文本表征直接切分。

为解决这一问题，我们将准则构建形式化为一个**“多视图采样 $\rightarrow$ 语义空间聚类分流 $\rightarrow$ 实证残差正交筛选”**的三阶段数学过程：

* **跨视图不变性导出“最小证据覆盖”**：分别基于纯指令 $(q)$ 以及指令配对不同候选回答 $(q, y_m)$ 生成多个版本的原子准则集合。将这些多视角准则投射到同一语义空间进行聚类后，**在多个视图中被高频共同生成的簇（Consensus Clusters）**天然对应了跨越具体回答的不变核心要求；而**每个簇仅提取唯一簇心代表元（Medoid）**，则在经验上去除了大部分同义重复。这只是**近似去重**：余弦阈值聚类不能识别蕴涵关系，可能把字面相近但语义相反的准则（如“≤100 词”与“≤200 词”）合并，也可能保留同义但措辞差异大的准则。
* **低频长尾簇结合实证方差导出“额外判别准则”**：仅在特定 $(q, y_m)$ 视图中被激发出来的低频簇，**可能**捕获特定回答的独有亮点或隐蔽缺陷，但也包含偶然噪声。v1 在探测得分矩阵上执行 Gram-Schmidt 正交残差投影，试图滤除无区分度的噪声；注意探测间的方差并不等于“区分质量好坏”的能力（见第 4 节）。

---

## 2. 数学形式化与方法建模 (Mathematical Formulation)

### 2.1 阶段一：多视图原子准则采样 (Multi-View Atomic Rubric Sampling)

给定用户指令 $q$ 以及一组包含 $M$ 个候选（或探测）响应的集合 $\mathcal{Y} = \{y_1, y_2, \dots, y_M\}$。我们构造 $M+1$ 个观测视图来激发原子评估准则：

1. **先验无偏视图（Prior View, $m=0$）**：仅输入指令 $q$，采样生成初始原子准则集：
   $$
   \mathcal{R}^{(0)} = \left\{ r_{0, 1}, r_{0, 2}, \dots, r_{0, n_0} \right\} \sim \mathbb{P}_{\mathcal{G}}(\mathcal{R} \mid q)
   $$
2. **响应锚定视图（Response-Grounded Views, $m=1, \dots, M$）**：依次将指令 $q$ 与第 $m$ 个候选回答 $y_m$ 配对（或将回答划分为若干子视图），引导生成模型审视 $y_m$ 的具体表现与潜在缺陷，独立生成第 $m$ 个版本的原子准则集：
   $$
   \mathcal{R}^{(m)} = \left\{ r_{m, 1}, r_{m, 2}, \dots, r_{m, n_m} \right\} \sim \mathbb{P}_{\mathcal{G}}(\mathcal{R} \mid q, y_m), \quad \forall m \in \{1, \dots, M\}
   $$

将全部 $M+1$ 个视图生成的原子准则取并集，构成原始多视图准则全集 $\mathcal{U}$：

$$
\mathcal{U} = \bigcup_{m=0}^M \mathcal{R}^{(m)}, \quad N = \vert{}\mathcal{U}\vert{} = \sum_{m=0}^M n_m
$$

---

### 2.2 阶段二：语义聚类分流与最小证据覆盖提取 (Semantic Clustering & Core-Residual Split)

原始全集 $\mathcal{U}$ 中包含大量跨视图重复的等价准则。设 $\phi: \mathcal{U} \to \mathbb{S}^{d-1}$ 为单位球面上的语义嵌入映射（或基于自然语言推理的等价核函数），定义准则 $r_i, r_j \in \mathcal{U}$ 之间的语义相似度为 $S_{\text{sem}}(r_i, r_j) = \langle \phi(r_i), \phi(r_j) \rangle$。

#### 1. 凝聚层次聚类与正交簇心提取

以相似度阈值 $\theta_{\text{sim}} \in (0, 1)$（如 $\theta_{\text{sim}} = 0.80$）对 $\mathcal{U}$ 执行凝聚层次聚类（Agglomerative Clustering），将 $\mathcal{U}$ 划分为 $C$ 个互不相交的语义等价簇 $\mathcal{G} = \{G_1, G_2, \dots, G_C\}$：

$$
\mathcal{U} = \bigsqcup_{c=1}^C G_c, \quad \text{满足 } \forall c \neq c', \ \text{Sim}(G_c, G_{c'}) < \theta_{\text{sim}}
$$

对于每个簇 $G_c$，选取簇内与其他成员平均语义相似度最高的准则作为该簇的**唯一代表元（Medoid Rubric）** $\bar{r}_c$：

$$
\bar{r}_c = \arg\max_{r \in G_c} \frac{1}{\vert{}G_c\vert{}} \sum_{r' \in G_c} \langle \phi(r), \phi(r') \rangle
$$

这一步将 $N$ 条混杂准则压缩为 $C$ 条经验上去重后的候选准则 $\{\bar{r}_1, \dots, \bar{r}_C\}$。

#### 2. 跨视图支持度计算与“核心-额外”集合分流

为了区分哪些准则是必选的“最小证据覆盖”，哪些是由特定回答激发的“额外候选准则”，我们定义簇 $G_c$ 的**跨视图共识支持度（Cross-View Consensus Support）** $\rho(G_c) \in [0, 1]$：

$$
\rho(G_c) = \beta \cdot \mathbb{I}\left(G_c \cap \mathcal{R}^{(0)} \neq \emptyset\right) + (1 - \beta) \cdot \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left(G_c \cap \mathcal{R}^{(m)} \neq \emptyset\right)
$$

其中 $\mathbb{I}(\cdot)$ 为示性函数，衡量该簇是否在第 $m$ 个视图中被独立提出；$\beta \in (0, 1)$ 为先验视图的锚定权重（默认取 $\beta = 0.4$）。

给定共识阈值 $\tau_{\text{core}}$（例如 $\tau_{\text{core}} = 0.5$），准则簇自然分流为两个互补的子集：

* **最小证据覆盖集（Minimal Evidence Coverage Set, $\mathcal{R}_{\text{core}}$）**：
  $$
  \mathcal{R}_{\text{core}} = \left\{ \bar{r}_c \ \middle\vert{}\ \rho(G_c) \ge \tau_{\text{core}}, \ c \in \{1, \dots, C\} \right\}
  $$

  *数学性质：* $\mathcal{R}_{\text{core}}$ 中的每一条准则要么同时被先验视图与响应视图双重确认，要么在绝大多数候选回答视图中均被独立提取为必要条件；每个簇只保留一个代表元 $\bar{r}_c$，因此 $\mathcal{R}_{\text{core}}$ 的规模受簇数约束；但覆盖完备性与最小性都**没有保证**。还需注意：在 $\beta=0.4,\tau_{\text{core}}=0.5,M=6$ 时，含先验成员的簇只需再被 1 个响应视图命中（$\rho=0.4+0.6/6=0.5$）即可入核，而不含先验成员的簇需要 $\ge 5/6$ 的响应视图支持，因此 core 在结构上接近“被回声一次的先验准则”。这一点需要用 `src/mccr_diagnostics.py` 的 `core_share_prior_echoed_once` 实测。
* **响应衍生候选池（Response-Triggered Candidate Pool, $\tilde{\mathcal{R}}_{\text{ext}}$）**：
  $$
  \tilde{\mathcal{R}}_{\text{ext}} = \left\{ \bar{r}_c \ \middle\vert{}\ \rho(G_c) < \tau_{\text{core}}, \ c \in \{1, \dots, C\} \right\}
  $$

  *数学性质：* $\tilde{\mathcal{R}}_{\text{ext}}$ 中的准则仅在少数特定回答视图 $(q, y_m)$ 中被激发，代表了针对特定回答亮点的拔高检验或针对局部缺陷的专门排查。

---

### 2.3 阶段三：基于实证得分矩阵的残差正交筛选 (Empirical Orthogonal Residual Filtering)

候选池 $\tilde{\mathcal{R}}_{\text{ext}}$ 中既包含高价值的判别准则，也混杂着个别视图产生的偶然噪声。我们通过候选响应集合 $\mathcal{Y} = \{y_1, \dots, y_M\}$ 上的**实证评估矩阵**来提纯真正的额外准则 $\mathcal{R}_{\text{ext}} \subset \tilde{\mathcal{R}}_{\text{ext}}$。

设 Judge 算子对响应 $y_m$ 在准则 $r$ 上的打分为 $f(y_m, r) \in [0, 1]$。
记 $\mathcal{R}_{\text{core}}$（共 $K$ 条）在 $\mathcal{Y}$ 上的得分矩阵为 $\mathbf{F}_{\text{core}} \in [0, 1]^{M \times K}$；记 $\tilde{\mathcal{R}}_{\text{ext}}$（共 $P$ 条）在 $\mathcal{Y}$ 上的得分矩阵为 $\mathbf{F}_{\text{ext}} \in [0, 1]^{M \times P}$。

#### 1. 列向量去均值中心化 (Centering)

定义中心化算子 $\mathbf{H} = \mathbf{I}_M - \frac{1}{M}\mathbf{1}_M\mathbf{1}_M^\top$，将得分矩阵转换为零均值判别剖面矩阵：

$$
\mathbf{V}_{\text{core}} = \mathbf{H}\mathbf{F}_{\text{core}}, \quad \mathbf{V}_{\text{ext}} = \mathbf{H}\mathbf{F}_{\text{ext}}
$$

此时，任意候选准则 $r_p \in \tilde{\mathcal{R}}_{\text{ext}}$ 对应的中心化列向量 $\tilde{\mathbf{v}}_p$ 的二范数平方等于 $M$ 倍的经验方差：$\frac{1}{M}\Vert{}\tilde{\mathbf{v}}_p\Vert{}_2^2 = \text{Var}_{y \in \mathcal{Y}}[f(y, r_p)]$。

#### 2. Gram-Schmidt 正交补空间投影与贪心选择

为确保选出的额外准则不仅具有高区分度（高方差），而且不会与最小证据覆盖集 $\mathcal{R}_{\text{core}}$ 或已选额外准则产生评分共线性（防止同一个优缺点被换个说法重复计分），我们提取 $\text{span}(\mathbf{V}_{\text{core}})$ 的标准正交基矩阵 $\mathcal{B}_0$（通过精简 SVD 分解获得）。

在第 $t = 1, \dots, L$ 步迭代中，对于每个未选中的候选准则 $r_p \in \tilde{\mathcal{R}}_{\text{ext}}$，将其中心化得分向量 $\tilde{\mathbf{v}}_p$ 投影至当前标准正交基 $\mathcal{B}_{t-1}$ 的**正交补空间**：

$$
\mathbf{e}_p^{(t-1)} = \left(\mathbf{I}_M - \mathcal{B}_{t-1}\mathcal{B}_{t-1}^\top\right)\tilde{\mathbf{v}}_p
$$

残差向量的均方模长 $\sigma_p^2(t) = \frac{1}{M}\Vert{}\mathbf{e}_p^{(t-1)}\Vert{}_2^2$ 度量了准则 $r_p$ 在剥离掉现有准则影响后，所能提供的**纯净未解释判别方差（Unexplained Discriminative Variance）**。

结合该簇的原始视图支持度权重 $\alpha_p = \max(\rho(G_p), \rho_{\min})$，每一步贪心选择残差能量最大的候选准则：

$$
p^* = \arg\max_{p} \ \alpha_p \cdot \frac{1}{M}\left\Vert{}\mathbf{e}_p^{(t-1)}\right\Vert{}_2^2
$$

若 $\frac{1}{M}\Vert{}\mathbf{e}_{p^*}^{(t-1)}\Vert{}_2^2 < \tau_{\text{var}}$（残差方差低于阈值，说明已无新的区分信息），则提前终止选择；否则将 $r_{p^*}$ 并入最终额外准则集 $\mathcal{R}_{\text{ext}}$，并将单位化残差方向 $\mathbf{u}_t = \mathbf{e}_{p^*}^{(t-1)} / \Vert{}\mathbf{e}_{p^*}^{(t-1)}\Vert{}_2$ 扩充进正交基 $\mathcal{B}_t = [\mathcal{B}_{t-1}, \mathbf{u}_t]$。

---

### 2.4 阶段四：主干门控分层评分聚合 (Core-Gated Hierarchical Scoring)

最终确定的评估准则集合为 $\mathcal{R}^* = \mathcal{R}_{\text{core}} \sqcup \mathcal{R}_{\text{ext}}$。为防止额外准则 $\mathcal{R}_{\text{ext}}$ 稀释最小证据覆盖集 $\mathcal{R}_{\text{core}}$ 的决定性地位，待测回答 $y$ 的总得分 $U(y \mid q)$ 采用**主干乘性门控聚合**：

1. **最小证据覆盖得分（Core Coverage Score）：**
   $$
   S_{\text{core}}(y) = \sum_{k=1}^K w_k^{\text{core}} f\left(y, r_k^{\text{core}}\right), \quad w_k^{\text{core}} = \frac{\rho(G_k)}{\sum_{j=1}^K \rho(G_j)}
   $$

   *(共识支持度越高的核心准则，在基础分中的占比越高。)*
2. **额外判别补充得分（Supplementary Discriminative Score）：**
   $$
   S_{\text{ext}}(y) = \sum_{l=1}^L w_l^{\text{ext}} f\left(y, r_l^{\text{ext}}\right), \quad w_l^{\text{ext}} = \frac{\sigma_l}{\sum_{j=1}^L \sigma_j + \epsilon}
   $$

   *(其中 $\sigma_l = \frac{1}{\sqrt{M}}\Vert{}\mathbf{e}_l^{(l-1)}\Vert{}_2$ 为第 $l$ 条额外准则入选时的残差标准差。)*
3. **门控合成总效用（Gated Total Utility）：**
   $$
   U(y \mid q) = S_{\text{core}}(y) + \eta \cdot \left[S_{\text{core}}(y)\right]^\gamma \cdot S_{\text{ext}}(y)
   $$

   其中 $\gamma \ge 1$ 为门控指数（默认 $\gamma = 2.0$），$\eta \in (0, 1]$ 为补充项缩放系数（默认 $\eta = 0.5$）。当回答未满足最小证据覆盖时（$S_{\text{core}}(y) \ll 1$），门控因子 $[S_{\text{core}}(y)]^\gamma$ 迅速衰减趋近于 0，阻断额外加分项的干扰；仅当回答充分满足最小证据覆盖（$S_{\text{core}}(y) \to 1$）时，$\mathcal{R}_{\text{ext}}$ 才全额介入以拉开高质量回答之间的档次差距。

---

## 3. 算法伪代码与核心实现逻辑

### 3.1 完整算法流程 (Algorithm Block)

```
算法 1: MCCR-Eval (多视图聚类共识与残差正交准则生成算法)
输入: 用户指令 q, 候选响应集 Y = {y_1, ..., y_M}, 
      语义聚类阈值 theta_sim=0.80, 核心共识阈值 tau_core=0.50, 先验锚定系数 beta=0.40,
      最大额外准则数 L=4, 残差方差阈值 tau_var=0.04, 门控参数 (gamma=2.0, eta=0.5)
输出: 最小证据覆盖集 R_core, 额外补充准则集 R_ext, 以及各响应最终得分 {U(y_1), ..., U(y_M)}

1. [阶段一：多视图原子准则生成]
   生成先验视图准则: R^(0) <- Prompt_Prior_Atomic(q)
   For m = 1 to M do:
       生成响应锚定视图准则: R^(m) <- Prompt_Response_Atomic(q, y_m)
   汇总全集 U = R^(0) U R^(1) U ... U R^(M)

2. [阶段二：语义聚类与主干-残差分流]
   计算 U 中每条准则的归一化语义向量 phi(r)。
   基于余弦距离 (1 - <phi(r_i), phi(r_j)>) 和距离阈值 (1 - theta_sim) 执行凝聚层次聚类，得到簇 {G_1, ..., G_C}。
   初始化 R_core = [], R_tilde_ext = []。
   For c = 1 to C do:
       提取簇心代表元: r_bar_c = argmax_{r in G_c} mean_{r' in G_c} <phi(r), phi(r')>
       计算跨视图支持度: rho_c = beta * I(G_c ∩ R^(0) != empty) + (1 - beta) * (1/M) * sum_{m=1}^M I(G_c ∩ R^(m) != empty)
       If rho_c >= tau_core then:
           将 (r_bar_c, rho_c) 加入最小证据覆盖集 R_core
       Else:
           将 (r_bar_c, rho_c) 加入额外候选池 R_tilde_ext

3. [阶段三：实证得分矩阵构建与正交残差筛选]
   使用裁判模型评估 Y 在 R_core 与 R_tilde_ext 上的得分 -> 得到矩阵 F_core in [0,1]^(M x K), F_ext in [0,1]^(M x P)。
   列中心化: V_core = H * F_core, V_ext = H * F_ext。
   对 V_core 执行 SVD 提取标准正交基 B。
   初始化最终额外集 R_ext = [], 权重 w_ext = []。
   For t = 1 to L do:
       For 每个未选中的候选 p in R_tilde_ext do:
           e_p = (I_M - B * B^T) * V_ext[:, p]
           var_p = ||e_p||_2^2 / M
           score_p = max(rho_p, 0.2) * var_p
       p* = argmax_p (score_p)
       If var_{p*} < tau_var then Break
       将 r_bar_{p*} 加入 R_ext，记录权重 sigma_{p*} = sqrt(var_{p*})
       B = [B, e_{p*} / ||e_{p*}||_2]

4. [阶段四：门控分层效用计算]
   按公式计算每个回答的 S_core(y_m)、S_ext(y_m) 与最终效用 U(y_m) = S_core(y_m) + eta * S_core(y_m)^gamma * S_ext(y_m)。
```


---

## 4. MCCR v2：方向性弱监督筛选与可证明的分层聚合（`mccr_v2`）

### 4.1 v1 的三个结构性问题

1. **秩约束。** 中心化后的探测矩阵秩 $\le M-1=5$。core 常有 3–6 条，$\mathrm{rank}(\mathbf{V}_{\text{core}})$ 往往已占满正交补，残差 $\mathbf{e}_p\approx 0$，导致 $L'=0$。此时 v1 退化为只用 core。
2. **方差 ≠ 判别力。** v1 的目标 $\alpha_p\|\mathbf{e}_p\|^2/M$ 是无监督的。探测由强、弱模型交替生成，方差最大的方向往往是两模型之间的风格、长度差异，与质量无关；而且同一条准则即使与质量**负相关**（例如偏好冗长），也会被选中。
3. **门控没有保证。** $U=S_{\text{core}}+\eta S_{\text{core}}^{\gamma}S_{\text{ext}}$ 中，补充项可以抵消 core 上的小缺陷，与“核心要求优先”的主张不一致。

### 4.2 视图生成（覆盖优先）

先验视图 $\mathcal{R}^{(0)}$ 和响应视图 $\mathcal{R}^{(m)}$ 都要求尽可能覆盖全部要求（显式约束、隐含前提、核心目标、格式/长度/语言、安全边界），每条停留在“要求”层级，同一要求的子要点合并，不写实现细节和具体答案，也**不规定条数**。覆盖由各视图负责，去重和分层由后续的聚类共识与方向筛选负责：前一阶段追求召回，后一阶段追求精度。

### 4.2.1 已知方向的探测对

设评分行集合为 $\mathcal{Y}=\{y_1,\dots,y_M\}\cup\{\tilde y_1,\dots,\tilde y_K\}$，其中 $\tilde y_k$ 是对强生成器探测 $y_{s(k)}$ 的**反事实降级**：由生成器改写，只注入**一个**实质缺陷（违反一条显式约束、篡改关键事实或计算、遗漏必要部分、错误地拒绝或顺从），其余长度、语气和格式保持不变（`PROMPT_COUNTERFACTUAL`）。由此得到“已知更好 ≻ 已知更差”的有序对集合

$$
\mathcal{P}=\underbrace{\{(i,j): y_i\in\mathcal{Y}_G,\ y_j\in\mathcal{Y}_W\}}_{\text{强≻弱（弱监督）}}\ \cup\ \underbrace{\{(s(k),\tilde k)\}}_{\text{原始≻反事实}} .
$$

反事实对控制了风格变量，因此对齐信号来自内容质量，而不是模型风格。

### 4.3 方向对齐与核心过滤

对每条 medoid 准则 $r$，定义方向对齐度

$$
a_r=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\bigl(f(y_i,r)-f(y_j,r)\bigr)\in[-1,1].
$$

$a_r<-\tau_{\text{align}}$ 的 core 准则判为**错向**，从 core 中移除（RRD 只用强/弱对做硬过滤，这里推广为连续量，并加入反事实对）。

### 4.4 补充准则的对比式选择

候选按 $a_p$ 降序贪心选择，入选条件为 $a_p\ge\tau_{\text{align}}$；另做共线性检验：若中心化得分向量 $\mathbf{v}_p$ 在当前基 $\mathcal{B}$ 正交补中的能量占比
$\|(\mathbf{I}-\mathcal{B}\mathcal{B}^\top)\mathbf{v}_p\|^2/\|\mathbf{v}_p\|^2<1-\kappa$（默认 $\kappa=0.9$），则视为与已有准则重复并跳过。共线性检验在行数很少时依然有意义，不像残差方差阈值那样会因秩饱和而失效。

### 4.5 权重与聚合

- **core 权重**：$\mathbf{w}\propto\hat\Sigma^{-1/2}\mathbf{1}$，其中 $\hat\Sigma$ 是 core 得分在全部评分行上的 Ledoit–Wolf 收缩协方差，负权重截断为 0 后归一化。这对应 RRD 的白化加权（WU）；收缩是必需的，因为 $M+K<|\mathcal{R}_{\text{core}}|$ 时样本协方差奇异。
- **补充权重**：与 $a_p$ 成正比。
- **词典序聚合**：
$$
U(y)=S_{\text{core}}(y)+c\cdot\frac{\min_k w_k}{2}\,S_{\text{ext}}(y),\qquad 0<c<1 .
$$
**命题。** 当准则分数取值于 $\{0,\tfrac12,1\}$ 时，若 $S_{\text{core}}(y)>S_{\text{core}}(y')$，则 $U(y)>U(y')$。
*证明。* core 分数差至少为 $\min_k w_k/2$，而补充项之差最多为 $c\min_k w_k/2<\min_k w_k/2$。∎
均匀权重下即为 PAOR 的 $c/(2K)$ 缺陷带。该命题依赖离散分数；本实现只使用离散分数（不使用 logprob）。

### 4.6 需要通过实验检验的假设

- **H1 覆盖。** 共识聚类应提高对专家准则的召回率（RubricBench 专家准则召回，`src/rubric_recall.py`）。
- **H2 核心优先。** 词典序聚合应在“表面好、核心错”的对抗样本上更稳健（LLMBar Adversarial、RM-Bench Hard）。
- **H3 方向性选择。** 对比式选择应优于残差方差选择、随机选择和全选（消融），且准确率提升不能仅靠增加调用（准确率–算力曲线）。

### 4.7 默认超参数

`theta_sim=0.80, tau_core=0.50, beta=0.40`（与 v1 相同），`N=8`（4 强 + 4 弱），`n_counterfactual=4, tau_align=0.05, kappa=0.9, c=0.8, L=4`。先验视图和响应视图**不设准则条数**（覆盖优先、要求级粒度，仅保留 50 条防失控上限），因此“条数”不是超参数。v2 去掉了 v1 的 `tau_var, rho_min, gamma, eta`，因此需要调的超参数少于 v1。所有超参数只在 RewardBench 2 / HelpSteer3 的 dev 划分上确定，之后在全部数据集上冻结使用。

---

## 5. MCCR-ME：把 rubric 构造为最小证据集合（`mccr_me`，当前主方法）

### 5.1 定义

设 $A(q)$ 为 query $q$ 的可接受回答集合，准则集 $E$ 的通过集为 $P(E)=\{y:\ y\models e,\ \forall e\in E\}$。称 $E$ 是 $q$ 的**最小证据集合**，如果它满足：

- **扎根**：每条 $e$ 要么是 $q$ 的显式要求（能给出 $q$ 中的原文片段），要么是完成 $q$ 所必需的隐含要求；
- **充分**：$P(E)\subseteq A(q)$；
- **必要（极小）**：$\forall e\in E,\ \exists\, w_e \in P(E\setminus\{e\})\setminus A(q)$，即存在只违反 $e$ 的**见证回答**，且它不可接受；
- **无冗余**：不存在被 $E\setminus\{e\}$ 蕴含的 $e$。

这对应逻辑解释中的极小充分条件（prime implicant / abductive explanation），也是 Reiter 诊断理论中“极小碰撞集”的对偶。**必要 ≠ 有区分度**：像“用英文回答”这类准则，所有探测都会满足，因此在探测上没有方差，但它仍然是必要的。所以必要性必须单独检验，不能用 v1 的方差或 v2 的方向对齐来代替。

### 5.2 query 视图与 query+response 视图的对比

query 视图 $R_q$ 要求为每条准则标注 `type` 和 `evidence`；`explicit` 的原文片段经代码校验确实出现在 $q$ 中，否则降为 `implicit`。与响应视图聚类对齐后，每个簇归入一类：

| 类别 | 条件 | 处理 |
|---|---|---|
| `explicit` | 簇内含经校验的显式 query 准则 | 直接入选 $E$（显式要求按定义即必要） |
| `both`（Q∩R） | query 视图与 ≥1 个响应视图都提出 | 直接入选（两个独立来源相互印证） |
| `query_implicit`（Q-only） | 只有 query 视图提出，且不是显式要求 | 做必要性检验 |
| `response_only`（R-only） | 只有响应视图提出 | 做必要性检验（检验“只有看到回答才暴露的隐含要求”） |

代表文本优先使用 query 视图中的措辞。每个 query 最多检验 `max_tests`（默认 12）条候选，`query_implicit` 优先，`response_only` 按提出它的视图数排序；超出预算的候选丢弃并记录。

### 5.3 必要性检验（默认：见证测试）

对每条待检候选 $c$，以第一条强生成器探测 $b$ 为底稿，由 G 改写出见证 $w_c$：只违反 $c$，其余准则保持满足，其余内容尽量不变。由 J 对 $(b, w_c)$ 做正反两序比较：**两次都判 $b$ 更好时，$c$ 才算必要**。如果 G 声明 $c$ 无法单独违反（`IMPOSSIBLE: <ids>`），说明 $c$ 被这些准则蕴含，判为不必要，并把这条信息提供给去冗余步骤。

可切换的估计器（`--me-necessity`），用于消融：
- `judgment`：J 直接做思想实验判断（每个 query 1 次调用）；
- `hitting_set`：用强≻弱、原始≻反事实这些有序对的冲突集求贪心最小碰撞集，并剔除错向准则（零生成调用；无法发现所有探测都满足的必要准则）；
- `none`：只做分类和去冗余。

### 5.4 去冗余

J 一次性列出集合中哪些准则被其余准则蕴含（`PROMPT_ENTAILMENT`），再并入见证阶段的 IMPOSSIBLE 信号。按优先级 `explicit` > `both` > `query_implicit` > `response_only` 从低到高做**线性删除极小化**：一条准则只有在它的蕴含来源都仍保留时才删除，因此互相蕴含的一对准则会保留优先级较高的那条，不会两条都被删掉。

### 5.5 充分性检验（一轮反例搜索，CEGIS 式）

G 写出对抗回答 $a$：满足 $E$ 的全部准则，但比 $b$ 明显更差。若 J 判定 $a$ 在 $E$ 上全部满分、且在正反两序中都输给 $b$，就说明 $E$ 不充分。此时由 G 从 $a$ 中提取遗漏的要求（1–3 条），这些新候选同样要通过必要性检验后才并入 $E$。可用 `--me-sufficiency off` 关闭。

### 5.6 评分

$E$ 中每条准则都是必要条件，因此采用均匀权重：$U(y)=\frac{1}{|E|}\sum_{e\in E} f(y,e)$，分数取离散值 {0, 0.5, 1}。平局走共享的正反两序 direct fallback。v2 的补充准则机制**不在** ME 中使用：最小证据集合本身就是 rubric，平局交给 fallback 处理。

### 5.7 成本

N=8 时：1（query 视图）+ 8（探测）+ 8（响应视图）+ 3·T（T ≤ 12 条被检候选）+ 1（去冗余）+ 约 4–8（充分性）+ R（目标评分）。也就是约 30+3T+R 次调用，再加 fallback。主表需要同时报告准确率–算力曲线，以及 `judgment`、`hitting_set` 两个低成本估计器的结果。
