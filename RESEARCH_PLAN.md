# 研究方案（定稿）

## 面对神经网络中的异常值，复杂硬件数据格式 vs. 正交变换辅以基础张量量化

---

## 核心命题

> 是应该走向以 MX/NVFP 为代表的**复杂硬件数据格式**，还是拥抱**简单基础数据格式（INT）+ 数学变换（HAD）**的算法软硬件协同设计？

本研究的差异化价值在于：以 PyRTL 微架构建模为基础，从**信息论、数值分析、硬件实现代价**三个维度同时评估，而非仅停留在模型精度层面。

---

## 技术栈定稿

### 高精度基准
- `FP32`, `BF16`

### 硬件原生格式派（Hardware-Native）
| 格式 | 位宽 | 关键特征 |
|---|---|---|
| NVFP4 | 4-bit | NVIDIA Blackwell E2M1 格式，无 Block Scale |
| MXFP4 | 4-bit | OCP 标准，Block=32 共享指数，E2M1 |
| MXFP8 | 8-bit | OCP 标准，Block=32 共享指数，E4M3/E5M2 |
| MXINT4 | 4-bit | OCP 标准整数，Block=32 共享 Scale |
| MXINT8 | 8-bit | OCP 标准整数，Block=32 共享 Scale |
| NF4 | 4-bit | QLoRA NormalFloat，为高斯分布信息论最优设计 |
| FP6 | 6-bit | E3M2，介于 FP4/FP8 的 Pareto 中间点 |

### 变换降维派（Transform-Based，所有算法必须可固化硬件）
| 方案 | 硬件可固化性 | 关键特征 |
|---|---|---|
| SmoothQuant + INT | 高（通道Scale可预计算） | 代数等价变换，零旋转开销 |
| HAD + INT (Per-Channel) | 高（结构化蝶形网络） | FWHT O(N log N) |
| HAD + LUT | 高 | 非线性映射通过查找表实现 |
| HAD + SQ-Format | 高（Gather/Scatter 硬件单元） | HAD 全局化 + 稀疏高精度存储 |
| 随机旋转矩阵 + INT | 中（矩阵固化为 ROM） | 精度上限参照，对比 HAD 的结构化代价 |
| TurboQuant + INT | 中（随机缩放可预计算） | 随机 ±1 缩放矩阵，比 HAD 更轻量 |

> **设计原则**：可学习旋转（SpinQuant）因训练依赖无法固化，不列入对比范围。LLM.int8() 因与 SQ-Format 同源，以 SQ-Format 为核心代表混合精度路线。

---

## 第一阶段：实验基线与技术栈确立

所有格式实现于 `formats/` 目录，统一接口：
```python
class QuantFormat:
    def quantize(self, x: np.ndarray, bits: int) -> np.ndarray
    def dequantize(self, q: np.ndarray) -> np.ndarray
    def encoding_overhead(self) -> dict  # bits/element for metadata
```

---

## 第二阶段：微观张量级与分布鲁棒性分析

### 测试分布矩阵（7类）

| 分布类型 | 参数配置 | 建模目标 |
|---|---|---|
| 高斯 N(0,1) | σ=1 | 理想基准 |
| Laplace(0, b) | b=0.5, 1.0, 2.0 | FFN 权重层 |
| Student-t(ν) | ν=3, 5, 10 | Transformer 激活长尾 |
| 双峰 Bimodal | μ₁=-3, μ₂=+3, σ=0.5 | Attention Softmax 输出 |
| 混合高斯（通道级异常值） | 99%×N(0,1) + 1%×N(0,σ²) σ=30,50,100 | LLM 系统性通道异常值 |
| 随机突刺 Spiky | N(0,1) + 0.1%的 10×/50×/100× 倍突变 | AWQ 场景 |
| Log-Normal | μ=0, σ=1,2 | SiLU/GELU 后激活 |

### 核心评估指标（5项）

| 指标 | 公式 | 意义 |
|---|---|---|
| MSE | E[(x - x̂)²] | 量化失真主指标 |
| SNR | 10·log₁₀(Var(x)/MSE) | 信噪比（dB） |
| KL 散度 | KL(P_x ‖ P_x̂) | 分布形态保持度 |
| Max-AE | max|x - x̂| | 单点异常值最坏情形 |
| 有效位宽 EffBits | -log₂(MSE/Var(x))/2 | 信息论等效编码位宽 |

---

## 第三阶段：位宽消融实验

### 8-bit 效率博弈（Efficiency Regime）
- 场景：W8A8
- 命题：既然 MXFP8 与基础 INT8 精度差异不再致命，HAD 变换带来的额外开销是否合理？
- 寻找精度饱和点：HAD 在 8-bit 下何时变成"画蛇添足"

### 4-bit 生存极限（Survival Regime）
- 场景：W4A4, W4A8
- 命题：异常值导致的精度坍塌，MX 的细粒度 Block Scale vs. HAD 的全局高斯化，谁更有效？
- 重点对比 MXFP4 vs. HAD+INT4 vs. SQ-Format 在极端异常值分布下的表现

---

## 第四阶段：基于 PyRTL 的硬件代价评估

### BOPs 计数
在推理代码中嵌入 Bit-Operations 计数器，记录每次前向传播的理论操作开销。

### PyRTL 微架构模块（全套）

| 模块 | 描述 | 方案归属 |
|---|---|---|
| `INT4/8 16×16 脉动阵列` | 纯整数 MAC，流水线寄存器 | 方案B基础 |
| `MXFP4/8 16×16 脉动阵列` | 含 Block Scale 广播、指数对齐移位器 | 方案A |
| `1D FWHT 蝶形变换` | N点快速Walsh-Hadamard，O(N log N) | 方案B预处理 |
| `SQ-Format Gather/Scatter` | 稀疏高精度 Gather + 密集低精度 Scatter | 方案B扩展 |
| `格式转换模块` | NF4/FP6/NVFP4 的编解码逻辑 | 各方案开销 |
| `SmoothQuant Scale乘法器` | 通道级缩放因子应用 | 方案B轻量版 |

### 格式转换开销（完整建模）
- MXFP：Block Scale 提取 + 广播开销
- FP6：非标准位宽打包/对齐开销
- NF4：LUT 查表解码延迟
- SQ-Format：稀疏掩码生成 + Gather 仲裁逻辑
- NVFP4：E2M1 编解码

### 硬件指标体系（全套）

| 指标 | 方法 | 说明 |
|---|---|---|
| 等效门数 Gate Count | `pyrtl.area_estimation()` + 解析模型校准 | 面积代理指标 |
| 关键路径延迟 | `pyrtl.timing_estimation()` | 决定最高时钟频率 |
| 能耗/OP (pJ) | Horowitz 45nm 解析模型 | 整数加法0.03pJ，内存读取200pJ/32bit |
| 内存带宽放大因子 | 元数据字节 / 数据字节 | MX Block Scale 的隐性开销 |
| 解码延迟（格式转换） | PyRTL 关键路径分析 | 是否在关键路径上 |
| 算术强度 (FLOPs/Byte) | 解析计算 | 用于 Roofline 定位 |

### 方案A vs 方案B 对比框架

```
方案A：复杂 MX 阵列
  MXFP4/8 MAC Array + Block Scale 广播 + 指数对齐

方案B：纯 INT 阵列 + HAD 预处理
  INT4/8 MAC Array + FWHT Module（面积被算力阵列分摊）

方案B+：INT 阵列 + HAD + SQ-Format
  INT4/8 MAC Array + FWHT + Gather/Scatter Unit
```

### Yosys 集成（可选，增强可信度）
PyRTL 支持导出 Verilog，若环境中安装 Yosys，则调用：
```bash
yosys -p "synth -top top; stat" output.v
```
提取 cell 数量与面积，与 PyRTL 估算结果相互印证。若 Yosys 不可用，自动回退到 PyRTL 估算 + 2.5× 校准系数。

---

## 第五阶段：核心图表清单（10类）

| # | 图表类型 | X 轴 | Y 轴 | 核心论点 |
|---|---|---|---|---|
| 1 | 分布演变 CDF + Q-Q Plot | 张量数值区间 | 累积概率 / 理论分位数 | HAD 与随机旋转如何将长尾分布强制高斯化 |
| 2 | 精度-异常值热力图 | 异常值倍数/比例 | 格式与技术栈 | 颜色=MSE；证明 MX Block Scale 在何种极端分布下失效 |
| 3 | 位宽-精度 Pareto 前沿（图A） | 等效位宽 (4/8-bit) | MSE / EffBits | 4-bit 生存区 vs. 8-bit 效率区的性价比之王 |
| 4 | 位宽-精度 Pareto 前沿（图B） | 等效位宽 (4/8-bit) | 内存带宽放大因子 | 揭示 MX 元数据的隐性带宽开销 |
| 5 | HAD vs. 随机旋转折线图 | 张量维度 N (64~4096) | MSE | 廉价结构化 HAD 在多大维度可媲美昂贵随机旋转 |
| 6 | 硬件 PPA 气泡图 | 解码延迟 (ns) | 端到端精度 (EffBits) | 气泡大小=能耗；一锤定音给出系统级最优解 |
| 7 | Roofline 模型图 | 算术强度 (FLOPs/Byte) | 理论峰值性能 (TOPs) | 各格式是否 Memory Bound；MX 元数据是否使其重返内存瓶颈 |
| 8 | 逐通道误差热力图 | 通道索引 (0→C) | 格式/技术栈 | 系统性通道异常值场景下各方案的局部救援能力 |
| 9 | 格式编码效率图 | 各格式 | 存储位宽 vs. 有效位宽 | 格式复杂度是否换来等价信息密度提升 |
| 10 | 硬件流水线延迟拆解图 | 流水线阶段 | 延迟 (ns) | HAD 开销是否真的被阵列算力分摊；MX 解码是否在关键路径 |

---

## 项目目录结构

```
dataformat/
├── RESEARCH_PLAN.md
├── requirements.txt
├── config.py                        # 全局配置与格式注册表
├── formats/
│   ├── baseline.py                  # FP32, BF16
│   ├── fp6.py                       # FP6 E3M2
│   ├── nf4.py                       # NF4 NormalFloat
│   ├── mxfp.py                      # MXFP4/8
│   ├── mxint.py                     # MXINT4/8
│   ├── nvfp4.py                     # NVFP4 E2M1
│   ├── sq_format.py                 # SQ-Format
│   └── transforms/
│       ├── hadamard.py              # FWHT
│       ├── random_rotation.py       # 随机正交旋转
│       └── smoothquant.py           # SmoothQuant
├── distributions/
│   ├── generators.py                # 7类分布
│   └── metrics.py                   # 5项指标
├── experiments/
│   ├── robustness.py                # 分布鲁棒性实验
│   └── bitwidth_ablation.py         # 位宽消融
├── hardware/
│   ├── bop_counter.py
│   ├── energy_model.py              # Horowitz 45nm
│   ├── roofline.py
│   ├── pyrtl_modules/
│   │   ├── int_mac_array.py
│   │   ├── mxfp_mac_array.py
│   │   ├── fwht_module.py
│   │   ├── sq_gather_scatter.py
│   │   └── format_converters.py
│   └── ppa_evaluator.py
└── visualization/
    ├── style.py
    ├── plot_distributions.py
    ├── plot_outlier_heatmap.py
    ├── plot_pareto.py
    ├── plot_had_vs_random.py
    ├── plot_ppa_bubble.py
    ├── plot_roofline.py
    ├── plot_channel_heatmap.py
    ├── plot_encoding_eff.py
    └── plot_pipeline.py
```
