# PaddleFleet 模块架构文档

## 项目概述

PaddleFleet 是一个专注于大规模分布式训练的核心功能库（Core Functional Library for Large Scale Distributed Training）。该项目基于 PaddlePaddle 框架，提供了高效的分布式训练能力，支持多种并行策略和优化技术。

## 核心特性

- **多种并行策略**：支持数据并行、张量并行、流水线并行等多种并行方式
- **高性能算子**：提供融合算子和自定义 CUDA 算子优化
- **大规模模型支持**：支持 GPT、ViT、多模态等多种模型架构
- **灵活的配置系统**：支持 YAML 配置和命令行参数
- **MoE (Mixture of Experts) 支持**：完整的专家混合模型实现

## 主要模块架构

```
paddlefleet/
├── models/              # 模型实现
│   ├── gpt/            # GPT 模型
│   ├── vision/         # 视觉模型 (ViT, CLIP)
│   ├── multimodal/     # 多模态模型 (LLaVA)
│   └── common/         # 通用模型组件
├── transformer/         # Transformer 核心组件
│   ├── moe/            # MoE (专家混合) 实现
│   ├── attention.py    # 注意力机制
│   ├── mlp.py          # 前馈网络
│   └── layer.py        # Transformer 层
├── tensor_parallel/     # 张量并行
├── pipeline_parallel/   # 流水线并行
├── distributed/         # 分布式训练工具
├── training/           # 训练初始化和配置
├── fusions/            # 融合算子
├── ops/                # 自定义算子
├── _extensions/        # C++/CUDA 扩展
└── refined_recompute/  # 精细化重计算
```

## 模块详细说明

### 1. Models 模块 (`models/`)

模型实现模块，提供了多种预训练模型架构的实现。

#### 1.1 GPT 模块 (`models/gpt/`)

**功能**：实现 GPT 系列大语言模型

**核心组件**：
- `GPTModel`: GPT 模型主类
- `GPTConfig`: GPT 模型配置
- `gpt_embedding.py`: 词嵌入层实现
- `gpt_layer_specs.py`: GPT 层规格定义
- `moe_layer_specs.py`: MoE 层规格定义
- `lm_head.py`: 语言模型头部

**特点**：
- 支持标准 GPT 架构和 MoE 变体
- 支持多种位置编码方式
- 集成张量并行和流水线并行
- 支持梯度检查点和重计算

#### 1.2 Vision 模块 (`models/vision/`)

**功能**：实现视觉模型

**核心组件**：
- `clip_vit_model.py`: CLIP ViT 模型实现
- `vit_layer_specs.py`: ViT 层规格定义
- `multimodal_projector.py`: 多模态投影器
- `radio.py`: RADIO 视觉编码器

**特点**：
- 支持 CLIP 风格的视觉-语言对齐
- 灵活的 ViT 架构配置
- 支持多种图像分辨率

#### 1.3 Multimodal 模块 (`models/multimodal/`)

**功能**：实现多模态模型

**核心组件**：
- `llava_model.py`: LLaVA 多模态模型
- `llava_spec.py`: LLaVA 规格定义
- `context_parallel.py`: 上下文并行支持

**特点**：
- 图像-文本多模态理解
- 支持多图像输入
- 优化的上下文并行策略

#### 1.4 Common 模块 (`models/common/`)

**功能**：提供模型通用组件

**核心组件**：
- `embeddings/`: 各种嵌入实现
  - `rotary_pos_embedding.py`: RoPE 位置编码
  - `yarn_rotary_pos_embedding.py`: YaRN RoPE
  - `language_model_embedding.py`: 语言模型嵌入
- `language_loss/`: 语言模型损失函数
- `vision_layer/`: 视觉层实现

### 2. Transformer 模块 (`transformer/`)

Transformer 核心组件的实现。

**核心组件**：
- `FleetLayer`: Fleet Transformer 层基类
- `TransformerConfig`: Transformer 配置类
- `attention.py`: 多头注意力机制
- `mlp.py`: 前馈神经网络
- `dot_product_attention.py`: 点积注意力
- `transformer_layer.py`: 完整的 Transformer 层
- `transformer_block.py`: Transformer 块

**特点**：
- 高度模块化和可配置
- 支持各种注意力变体（Flash Attention、Block Mask 等）
- 支持 Pre-LN 和 Post-LN
- 集成张量并行支持

#### 2.1 MoE 子模块 (`transformer/moe/`)

**功能**：实现专家混合（Mixture of Experts）架构

**核心组件**：
- `moe_layer.py`: MoE 层实现
- `moe_router.py`: 专家路由器
- `moe_expert.py`: 专家网络
- `moe_shared_expert.py`: 共享专家
- `token_dispatcher.py`: Token 分发器
- `fused_a2a.py`: 融合的 All-to-All 通信
- `fp8_utils.py`: FP8 量化工具
- `moe_utils.py`: MoE 辅助函数

**特点**：
- 支持 Top-K 路由和容量限制
- 支持专家并行
- 优化的 Token 分发和聚合
- 支持 FP8 量化加速
- 融合的通信算子

### 3. Tensor Parallel 模块 (`tensor_parallel/`)

张量并行实现，用于在多个设备上切分模型参数。

**核心组件**：
- `layers.py`: 并行化的线性层和嵌入层
  - `ColumnParallelLinear`: 列并行线性层
  - `RowParallelLinear`: 行并行线性层
  - `VocabParallelEmbedding`: 词表并行嵌入层
- `random.py`: 随机数管理和检查点
- `cross_entropy.py`: 并行化的交叉熵损失
- `mappings.py`: 张量并行映射操作
- `data.py`: 数据并行工具

**特点**：
- 支持 Megatron 风格的张量并行
- 自动处理前向和反向传播的通信
- 支持序列并行
- 高效的梯度同步

### 4. Pipeline Parallel 模块 (`pipeline_parallel/`)

流水线并行实现，支持多种流水线调度策略。

**核心组件**：
- `PipelineParallel`: 标准流水线并行
- `PipelineParallelWithInterleave`: 交错流水线并行（VPP）
- `PipelineParallelWithInterleaveFthenB`: F-then-B 调度策略
- `VPPFhenBInBalancedMemory`: 内存平衡的 VPP
- `pp_layers.py`: 流水线层定义
- `pipeline_hooks.py`: 流水线钩子
- `vpp_simulator.py`: VPP 模拟器

**子模块 (`pp_utils/`)**：
- `p2p_communication.py`: 点对点通信
- `four_directions_p2p_communication.py`: 四向 P2P 通信
- `forward_backward_overlap_utils.py`: 前向反向重叠工具

**特点**：
- 支持 1F1B、GPipe 等多种调度策略
- 支持虚拟流水线并行（VPP）
- 优化的通信重叠
- 灵活的层分割策略
- 内存优化的调度

### 5. Distributed 模块 (`distributed/`)

分布式训练工具和模型包装器。

**核心组件**：
- `model.py`: 分布式模型包装

**特点**：
- 统一的分布式训练接口
- 支持多种并行策略组合
- 自动化的通信管理

### 6. Training 模块 (`training/`)

训练初始化和配置管理。

**核心组件**：
- `initialize.py`: 初始化函数 `initialize_fleet`
- `arguments.py`: 命令行参数定义
- `yaml_arguments.py`: YAML 配置支持
- `global_vars.py`: 全局变量管理

**特点**：
- 简化的训练初始化流程
- 支持 YAML 配置文件
- 灵活的参数系统
- 分布式环境自动配置

### 7. Fusions 模块 (`fusions/`)

融合算子实现，提高计算效率。

**核心组件**：
- `fused_layer_norm.py`: 融合的 Layer Normalization
- `fused_rms_norm.py`: 融合的 RMS Normalization
- `fused_softmax.py`: 融合的 Softmax
- `fused_bias_gelu.py`: 融合的 Bias + GELU
- `fused_bias_swiglu.py`: 融合的 Bias + SwiGLU
- `fused_bias_geglu.py`: 融合的 Bias + GEGLU
- `fused_bias_dropout.py`: 融合的 Bias + Dropout

**特点**：
- 减少内存访问
- 提高计算效率
- 支持多种激活函数融合
- 优化的反向传播

### 8. Ops 模块 (`ops/`)

自定义算子和第三方库集成。

**核心组件**：
- `deep_gemm`: 深度 GEMM 算子（需要 Hopper GPU）
- `deep_ep`: 深度专家并行（需要 Hopper GPU）
- `sonicmoe`: SonicMoE 优化（需要 Python 3.12+, CUDA 12.9+, Hopper GPU）
- `utils.py`: 算子工具函数

**特点**：
- 硬件感知的算子加载
- 支持 Triton 算子
- 动态库加载管理
- 版本和硬件兼容性检查

### 9. Extensions 模块 (`_extensions/`)

C++/CUDA 扩展实现。

**核心组件**：
- 14 个 CUDA 算子文件 (.cu)
- `flashmask/`: Flash Attention 与 Block Mask 支持
  - `rr_attn_estimate_triton_op.py`: RR Attention 估计（Triton）
  - `block_mask_utils.py`: Block Mask 工具
  - `index_utils.py`: 索引工具

**CUDA 算子列表**：
- `fuse_transpose_split_fp8_quant.cu`: FP8 量化转置分割
- `tokens_stable_unzip.cu`: Token 稳定解压
- `tokens_unzip_gather.cu`: Token 解压聚合
- `tokens_zip_unique_add.cu`: Token 压缩唯一添加
- `tokens_zip_prob.cu`: Token 压缩概率
- `merge_subbatch_cast.cu`: 子批次合并类型转换
- `tokens_unzip_slice.cu`: Token 解压切片
- `fuse_swiglu_scale.cu`: SwiGLU 融合缩放
- `swiglu_kernel.cu`: SwiGLU 核心
- `fuse_weighted_swiglu_fp8_quant.cu`: 加权 SwiGLU FP8 量化
- `router_metadata.cu`: 路由器元数据
- `count_cumsum.cu`: 计数累积和
- `filter_scores.cu`: 过滤分数
- `fuse_stack_transpose_fp8_quant.cu`: 堆叠转置 FP8 量化

**特点**：
- 高性能的自定义算子
- 支持 FP8 量化
- 优化的 MoE Token 处理
- Flash Attention 集成

### 10. Refined Recompute 模块 (`refined_recompute/`)

精细化重计算（梯度检查点）实现。

**核心组件**：
- `flash_attn.py`: Flash Attention 重计算支持
- `queue_check.py`: 队列检查工具

**特点**：
- 选择性重计算
- 减少内存占用
- 优化的反向传播
- 与 Flash Attention 集成

### 11. 其他核心模块

#### 11.1 `parallel_state.py`

**功能**：管理并行状态和进程组

**特点**：
- 全局并行状态管理
- 进程组初始化和管理
- 通信组获取

#### 11.2 `model_parallel_config.py`

**功能**：模型并行配置

**特点**：
- 统一的并行配置接口
- 支持多维并行

#### 11.3 `context_parallel_utils.py`

**功能**：上下文并行工具

**特点**：
- 长序列并行支持
- 序列维度切分

#### 11.4 `timers.py`

**功能**：性能计时器

**特点**：
- 分层计时
- 性能分析支持

#### 11.5 `spec_utils.py`

**功能**：层规格工具

**特点**：
- `LayerSpec`: 层规格定义类
- 支持延迟实例化

## 技术特点

### 1. 多维并行策略

PaddleFleet 支持以下并行策略的灵活组合：

- **数据并行（Data Parallel, DP）**：在多个设备上复制模型，每个设备处理不同的数据批次
- **张量并行（Tensor Parallel, TP）**：将模型参数按张量维度切分到多个设备
- **流水线并行（Pipeline Parallel, PP）**：将模型按层切分到多个设备，形成流水线
- **序列并行（Sequence Parallel, SP）**：将序列维度切分到多个设备
- **上下文并行（Context Parallel, CP）**：针对长序列的并行策略
- **专家并行（Expert Parallel, EP）**：MoE 模型中专家的并行

### 2. 高性能优化

- **融合算子**：减少 kernel 启动开销和内存访问
- **FP8 量化**：支持混合精度训练
- **Flash Attention**：优化的注意力计算
- **梯度累积**：支持大批次训练
- **梯度检查点**：减少内存占用

### 3. 灵活的配置系统

- YAML 配置文件支持
- 命令行参数覆盖
- 动态配置加载
- 配置验证和默认值

### 4. 扩展性设计

- 模块化架构
- 可插拔的组件
- 自定义算子支持
- 第三方库集成

## 依赖关系

### 核心依赖

- PaddlePaddle >= 3.0（GPU 或 XPU 版本）
- Python >= 3.10

### 可选依赖

- Triton（用于 deep_gemm 和 flashmask）
- nvidia-cutlass-dsl（用于 SonicMoE）
- NVSHMEM（用于 deep_ep）

### 硬件要求

- **基础功能**：支持 NVIDIA GPU (CUDA) 和昆仑芯 XPU
- **Deep GEMM / Deep EP**：需要 Hopper 架构 GPU（计算能力 >= 9.0）
- **SonicMoE**：需要 Hopper 架构 GPU + Python 3.12+ + CUDA 12.9+

## 使用场景

PaddleFleet 适用于以下场景：

1. **大规模语言模型训练**：GPT、LLaMA 等模型的预训练和微调
2. **视觉模型训练**：ViT、CLIP 等视觉模型
3. **多模态模型训练**：LLaVA 等图文多模态模型
4. **MoE 模型训练**：专家混合架构的大规模训练
5. **长序列训练**：支持超长上下文的模型训练
6. **多维并行实验**：测试和优化不同的并行策略组合

## 总结

PaddleFleet 是一个功能强大、高度模块化的分布式训练框架。它提供了：

- 完整的并行策略支持（DP、TP、PP、EP、SP、CP）
- 丰富的模型实现（GPT、ViT、LLaVA、MoE）
- 高性能的优化（融合算子、FP8、Flash Attention）
- 灵活的配置和扩展能力
- 硬件感知的算子加载

无论是研究人员进行模型实验，还是工程师进行大规模生产训练，PaddleFleet 都能提供强大的支持。
