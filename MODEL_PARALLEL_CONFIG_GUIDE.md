# ModelParallelConfig 配置指南

## 概述

`ModelParallelConfig` 是 PaddleFleet 中最核心的配置类，定义在 `src/paddlefleet/model_parallel_config.py` 文件中。这个数据类（dataclass）包含了所有与模型并行、训练优化、流水线并行等相关的配置参数，是构建大规模分布式训练系统的基础。

## 文件位置

```
src/paddlefleet/model_parallel_config.py
```

## 基本用法

```python
from paddlefleet.model_parallel_config import ModelParallelConfig

# 创建基本配置
config = ModelParallelConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    bf16=True,
)

# 在训练初始化中使用
from paddlefleet.training import initialize_fleet
initialize_fleet(model_parallel_config=config)
```

## 配置参数详解

### 1. 模型并行配置（Model Parallelism）

#### 1.1 张量并行（Tensor Parallelism）

**`tensor_model_parallel_size: int = 1`**
- **功能**：层内模型并行度，将张量按维度切分到多个 GPU
- **默认值**：1（不使用张量并行）
- **推荐设置**：
  - 小模型（< 10B）：1-2
  - 中等模型（10B-70B）：4-8
  - 大模型（> 70B）：8-16
- **示例**：
  ```python
  config = ModelParallelConfig(tensor_model_parallel_size=8)
  ```

**`parallel_output: bool = True`**
- **功能**：是否保持输出在张量并行的各个 rank 上分散（不聚合）
- **默认值**：True
- **说明**：
  - True：输出保持分散状态，节省通信和内存
  - False：将输出聚合到所有 rank
- **使用场景**：通常保持为 True，除非后续操作需要完整的输出

**`expert_tensor_parallel_size: int = None`**
- **功能**：MoE 专家层的张量并行度
- **默认值**：None（自动设置为与 `tensor_model_parallel_size` 相同）
- **说明**：允许专家层使用不同的张量并行度
- **示例**：
  ```python
  config = ModelParallelConfig(
      tensor_model_parallel_size=8,
      expert_tensor_parallel_size=4,  # 专家层使用较小的 TP
  )
  ```

#### 1.2 流水线并行（Pipeline Parallelism）

**`pipeline_model_parallel_size: int = 1`**
- **功能**：层间模型并行度，将 Transformer 层切分到多个 GPU
- **默认值**：1（不使用流水线并行）
- **推荐设置**：
  - 根据层数和可用节点数决定
  - 常见值：2, 4, 8, 16
- **示例**：
  ```python
  config = ModelParallelConfig(pipeline_model_parallel_size=4)
  ```

**`pipeline_model_parallel_comm_backend: str = None`**
- **功能**：配置流水线并行的通信后端
- **默认值**：None（使用默认后端）
- **可选值**：`"nccl"`, `"ucc"` 等
- **示例**：
  ```python
  config = ModelParallelConfig(
      pipeline_model_parallel_comm_backend="nccl"
  )
  ```

**`pipeline_model_parallel_split_rank: int = None`**
- **功能**：在 encoder-decoder 模型（如 T5）中，指定 encoder 和 decoder 的分割点
- **默认值**：None（不分割）
- **说明**：指定哪个 rank 是 encoder 和 decoder 的分界点

#### 1.3 虚拟流水线并行（Virtual Pipeline Parallelism）

**`virtual_pipeline_model_parallel_size: int = None`**
- **功能**：虚拟流水线并行度，用于减少流水线气泡
- **默认值**：None（不使用虚拟流水线并行）
- **工作原理**：
  - 将每个 Transformer 块进一步切分为更小的虚拟块
  - 每个流水线 rank 包含多个虚拟块
  - 通过交错执行减少气泡
- **推荐设置**：2-4（流水线并行度的倍数）
- **参考论文**：[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/pdf/2104.04473.pdf)
- **示例**：
  ```python
  config = ModelParallelConfig(
      pipeline_model_parallel_size=8,
      virtual_pipeline_model_parallel_size=2,  # 每个 PP rank 有 2 个虚拟块
  )
  ```

**`microbatch_group_size_per_vp_stage: int = None`**
- **功能**：每个虚拟流水线阶段一次执行的 micro-batch 数量
- **默认值**：None（自动设置为 `pipeline_model_parallel_size`，深度优先调度）
- **说明**：控制虚拟流水线的调度策略
- **示例**：
  ```python
  # PP=2, VP=2, microbatch_group_size=2, num_microbatches=4
  # rank 0: 0 1 0 1 2 3 2 3
  # rank 1:   0 1 0 1 2 3 2 3
  config = ModelParallelConfig(
      pipeline_model_parallel_size=2,
      virtual_pipeline_model_parallel_size=2,
      microbatch_group_size_per_vp_stage=2,
  )
  ```

#### 1.4 序列并行（Sequence Parallelism）

**`sequence_parallel: bool = False`**
- **功能**：启用序列并行，沿序列维度并行化 LayerNorm 和 Dropout
- **默认值**：False
- **要求**：必须启用张量并行（`tensor_model_parallel_size > 1`）
- **优势**：
  - 显著减少激活值内存占用
  - 对大模型（20B+）特别有效
- **参考论文**：[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- **示例**：
  ```python
  config = ModelParallelConfig(
      tensor_model_parallel_size=8,
      sequence_parallel=True,  # 需要 TP > 1
  )
  ```

#### 1.5 上下文并行（Context Parallelism）

**`context_parallel_size: int = 1`**
- **功能**：沿序列维度切分网络输入
- **默认值**：1（不使用上下文并行）
- **使用场景**：处理超长序列（如 32K、128K tokens）
- **示例**：
  ```python
  config = ModelParallelConfig(
      context_parallel_size=4,  # 将序列切分到 4 个 GPU
  )
  ```

**`hierarchical_context_parallel_sizes: list[int] = None`**
- **功能**：分层上下文并行的各级别大小
- **默认值**：None
- **说明**：
  - 提供一个列表来指定不同级别的大小
  - 例如 a2a+p2p 通信类型：第一个值是 a2a 通信组大小，第二个值是 p2p 通信组大小
- **示例**：
  ```python
  config = ModelParallelConfig(
      context_parallel_size=8,
      hierarchical_context_parallel_sizes=[4, 2],  # a2a=4, p2p=2
  )
  ```

#### 1.6 专家并行（Expert Parallelism）

**`expert_model_parallel_size: int = 1`**
- **功能**：MoE 专家分布的并行度
- **默认值**：1（不使用专家并行）
- **说明**：在数据并行的子维度上分布专家
- **要求**：与张量并行同时使用时必须启用序列并行
- **示例**：
  ```python
  config = ModelParallelConfig(
      expert_model_parallel_size=8,  # 8 个专家并行组
      tensor_model_parallel_size=4,
      sequence_parallel=True,  # 必需
  )
  ```

**`moe_extended_tp: bool = False`**
- **功能**：已废弃的标志（从 MCore v0.10 开始）
- **说明**：功能已被 `expert_tensor_parallel_size` 替代

### 2. 初始化配置（Initialization）

**`perform_initialization: bool = True`**
- **功能**：是否初始化权重
- **默认值**：True
- **使用场景**：如果要从检查点加载权重，可以设为 False 节省时间
- **示例**：
  ```python
  config = ModelParallelConfig(
      perform_initialization=False,  # 将从检查点加载
  )
  ```

**`use_cpu_initialization: bool = False`**
- **功能**：是否在 CPU 上初始化权重
- **默认值**：False（在 GPU 上初始化）
- **说明**：
  - False：直接在 GPU 上初始化（更快）
  - True：在 CPU 上初始化，然后传输到 GPU（对大模型可能很慢）
- **注意**：CPU 初始化对所有张量并行 rank 是相同的，GPU 初始化不同

### 3. 训练配置（Training）

#### 3.1 精度配置

**`fp16: bool = False`**
- **功能**：使用 FP16 混合精度训练
- **默认值**：False
- **注意**：与 `bf16` 互斥，只能选择一个

**`bf16: bool = False`**
- **功能**：使用 BF16 混合精度训练
- **默认值**：False
- **推荐**：对于现代 GPU（Ampere 及以上），推荐使用 BF16
- **示例**：
  ```python
  config = ModelParallelConfig(bf16=True)
  ```

**`params_dtype: paddle.dtype = paddle.float32`**
- **功能**：初始化权重时使用的数据类型
- **默认值**：`paddle.float32`
- **示例**：
  ```python
  import paddle
  config = ModelParallelConfig(
      params_dtype=paddle.bfloat16,
  )
  ```

**`enable_autocast: bool = False`**
- **功能**：在前向传播中启用 `paddle.amp.autocast` 上下文
- **默认值**：False

**`autocast_dtype: paddle.dtype = None`**
- **功能**：传递给 `paddle.amp.autocast` 的数据类型
- **默认值**：None（自动设置为 `params_dtype`）

#### 3.2 训练辅助函数

**`timers: Callable = None`**
- **功能**：计时器对象，用于各种计时功能
- **类型**：`paddlefleet.timers.Timers` 对象
- **示例**：
  ```python
  from paddlefleet import Timers
  config = ModelParallelConfig(timers=Timers())
  ```

**`finalize_model_grads_func: Callable = None`**
- **功能**：在所有 worker 上完成梯度的函数
- **说明**：可能包括确保梯度在 DP、PP、SP 维度上进行 all-reduce

**`grad_scale_func: Callable = None`**
- **功能**：损失缩放函数
- **说明**：如果使用损失缩放，此函数接受损失并返回缩放后的损失

**`no_sync_func: Callable = None`**
- **功能**：创建抑制异步数据并行通信的上下文的函数
- **说明**：如果模型是 `DistributedDataParallel` 实例，默认使用其 `no_sync` 方法

**`grad_sync_func: Callable = None`**
- **功能**：启动异步梯度归约的函数
- **说明**：例如分布式优化器的梯度 reduce-scatter

**`param_sync_func: Callable = None`**
- **功能**：启动异步参数同步的函数
- **说明**：例如分布式优化器的参数 all-gather

**`deterministic_mode: bool = False`**
- **功能**：启用确定性执行模式
- **默认值**：False
- **说明**：
  - True：选择确定性执行的代码（通常较慢）
  - False：允许非确定性优化
  - 适用于调试和测试

**`num_microbatches_with_partial_activation_checkpoints: int = None`**
- **功能**：设置部分层进行激活检查点的 micro-batch 数量
- **默认值**：None
- **说明**：
  - 如果设置为整数，指定数量的 micro-batch 不会对所有层进行检查点
  - 其余 micro-batch 将重计算所有层

**`fa_version: int = 2`**
- **功能**：FlashAttention 版本
- **默认值**：2
- **可选值**：2 或 3
- **说明**：也控制 Flashmask 的版本

### 4. 优化配置（Optimizations）

#### 4.1 梯度融合

**`gradient_accumulation_fusion: bool = False`**
- **功能**：将权重梯度累积融合到 GEMM 中
- **默认值**：False
- **要求**：需要自定义 CUDA 扩展 `fused_weight_gradient_mlp_cuda`
- **注意**：需要 CUDA >= 11

**`cross_entropy_loss_fusion: bool = False`**
- **功能**：使用融合的交叉熵实现
- **默认值**：False

**`cross_entropy_fusion_impl: str = "native"`**
- **功能**：交叉熵融合实现方式
- **默认值**：`"native"`
- **可选值**：
  - `"native"`：基于 MCore 的 CE loss fusion
  - `"te"`：Transformer Engine 库的并行 CE loss

#### 4.2 张量并行通信重叠

**`tp_comm_overlap: bool = False`**
- **功能**：启用线性层执行与张量并行通信的重叠
- **默认值**：False
- **说明**：在前向和反向传播中尽可能重叠 AllGather/ReduceScatter 通信

**`tp_comm_bulk_wgrad: bool = True`**
- **功能**：允许 All-Gather 与反向传播激活梯度 GEMM 重叠
- **默认值**：True
- **前提**：`tp_comm_overlap=True`

**`tp_comm_bulk_dgrad: bool = True`**
- **功能**：允许 Reduce-Scatter 与反向传播权重梯度 GEMM 重叠
- **默认值**：True
- **前提**：`tp_comm_overlap=True`

**`tp_comm_overlap_ag: bool = True`**
- **功能**：通过流水线化 GEMM 和 All-Gather 来实现重叠
- **默认值**：True
- **前提**：`tp_comm_overlap=True`

**`tp_comm_overlap_rs: bool = True`**
- **功能**：通过流水线化 GEMM 和 Reduce-Scatter 来实现重叠
- **默认值**：True
- **前提**：`tp_comm_overlap=True`

**`tp_comm_overlap_rs_dgrad: bool = False`**
- **功能**：允许 Reduce-Scatter 与 DGRAD GEMM 重叠
- **默认值**：False
- **前提**：`tp_comm_overlap=True`

**`tp_comm_overlap_disable_qkv: bool = False`**
- **功能**：禁用 QKV 层的 AllGather -> GEMM 重叠
- **默认值**：False

**`tp_comm_overlap_disable_fc1: bool = False`**
- **功能**：禁用 MLP FC1 层的 AllGather -> GEMM 重叠
- **默认值**：False

**`tp_comm_bootstrap_backend: str = "nccl"`**
- **功能**：设置引导通信后端
- **默认值**：`"nccl"`
- **可选值**：`"nccl"`, `"mpi"`, `"gloo"`

#### 4.3 废弃的通信重叠参数

以下参数从 TransformerEngine v1.6.0 开始已废弃：

- `async_tensor_model_parallel_allreduce: bool = False` - 已忽略
- `tp_comm_split_ag: bool = True` - 已废弃
- `tp_comm_atomic_ag: bool = False` - 已废弃
- `tp_comm_split_rs: bool = True` - 已废弃
- `tp_comm_atomic_rs: bool = False` - 已废弃

**`use_te_rng_tracker: bool = False`**
- **功能**：如果 TransformerEngine 存在，使用其 RNG 状态跟踪器
- **默认值**：False

### 5. 流水线并行配置（Pipeline Parallel）

#### 5.1 基本配置

**`variable_seq_lengths: bool = False`**
- **功能**：支持不同 micro-batch 之间的可变序列长度
- **默认值**：False
- **说明**：
  - 启用后会在流水线通信期间传递张量大小
  - 由于额外开销，仅在序列长度确实变化时使用

**`overlap_p2p_comm: bool = False`**
- **功能**：将部分点对点通信与计算重叠
- **默认值**：False
- **约束**：与 `batch_p2p_comm` 互斥（不能同时为 True）

**`batch_p2p_comm: bool = True`**
- **功能**：使用 `batch_isend_irecv` 而不是单独的 `isend/irecv` 调用
- **默认值**：True
- **约束**：与 `overlap_p2p_comm` 互斥

**`batch_p2p_sync: bool = True`**
- **功能**：使用 `batch_isend_irecv` 后执行 `cuda.device.synchronize()`
- **默认值**：True

**`overlap_p2p_comm_warmup_flush: bool = False`**
- **功能**：在预热和刷新阶段重叠通信和计算
- **默认值**：False
- **要求**：`overlap_p2p_comm=True` 且 `batch_p2p_comm=False`

**`deallocate_pipeline_outputs: bool = False`**
- **功能**：将输出数据发送到下一个流水线阶段后释放内存
- **默认值**：False
- **说明**：有助于节省内存，在不使用流水线并行时无效

#### 5.2 嵌入层权重梯度延迟

**`defer_embedding_wgrad_compute: bool = False`**
- **功能**：在流水线刷新期间延迟嵌入层的权重梯度 GEMM
- **默认值**：False
- **优势**：可以隐藏流水线刷新延迟
- **要求**：
  - 必须使用流水线并行（`pipeline_model_parallel_size > 1`）
  - 必须启用梯度累积融合（`gradient_accumulation_fusion=True`）

**`wgrad_deferral_limit: int = 0`**
- **功能**：需要延迟嵌入权重梯度计算的 micro-batch 数量
- **默认值**：0（延迟所有 micro-batch）
- **说明**：当 `defer_embedding_wgrad_compute=False` 时无效

**`delay_wgrad_compute: bool = False`**
- **功能**：延迟权重梯度计算以在组合 1F1B 中实现更好的重叠
- **默认值**：False

### 6. CPU Offloading 配置

**`cpu_offloading: bool = False`**
- **功能**：将所有激活异步卸载到 CPU
- **默认值**：False

**`cpu_offloading_num_layers: int = 0`**
- **功能**：指定需要卸载激活的 Transformer 层数量
- **默认值**：0

**`cpu_offloading_activations: bool = True`**
- **功能**：是否卸载激活到 CPU
- **默认值**：True

**`cpu_offloading_weights: bool = True`**
- **功能**：是否卸载权重到 CPU
- **默认值**：True

**`_cpu_offloading_context: AbstractContextManager = None`**
- **功能**：仅供内部使用，不要设置
- **说明**：将来会移动到"正确"的位置

### 7. 计时配置（Timing）

**`barrier_with_L1_time: bool = True`**
- **功能**：在 1 级时间测量中使用屏障
- **默认值**：True
- **警告**：用户需要确保调用屏障不会导致死锁
- **说明**：如果某些 rank 没有调用 1 级计时器，可能会发生挂起

## 配置验证（__post_init__）

`ModelParallelConfig` 在初始化后会执行验证逻辑（`__post_init__` 方法）：

### 自动调整

1. **序列并行自动关闭**：
   - 如果 `tensor_model_parallel_size <= 1`，自动将 `sequence_parallel` 设为 False

2. **专家张量并行默认值**：
   - 如果 `expert_tensor_parallel_size` 为 None，自动设置为 `tensor_model_parallel_size`

3. **自动转换数据类型**：
   - 如果 `autocast_dtype` 为 None，自动设置为 `params_dtype`

4. **虚拟流水线批次组大小默认值**：
   - 如果 `microbatch_group_size_per_vp_stage` 为 None，自动设置为 `pipeline_model_parallel_size`

### 配置约束检查

1. **序列并行要求**：
   ```python
   # 错误：不能在没有张量并行的情况下使用序列并行
   config = ModelParallelConfig(
       tensor_model_parallel_size=1,
       sequence_parallel=True,  # ❌ 会抛出 ValueError
   )
   ```

2. **专家并行 + 张量并行要求**：
   ```python
   # 错误：同时使用 EP 和 TP 时必须启用 SP
   config = ModelParallelConfig(
       expert_model_parallel_size=8,
       tensor_model_parallel_size=4,
       sequence_parallel=False,  # ❌ 会抛出 ValueError
   )
   ```

3. **嵌入权重梯度延迟要求**：
   ```python
   # 错误：延迟嵌入 wgrad 需要流水线并行
   config = ModelParallelConfig(
       defer_embedding_wgrad_compute=True,
       pipeline_model_parallel_size=1,  # ❌ 会抛出 ValueError
   )
   
   # 错误：延迟嵌入 wgrad 需要梯度累积融合
   config = ModelParallelConfig(
       defer_embedding_wgrad_compute=True,
       gradient_accumulation_fusion=False,  # ❌ 会抛出 ValueError
   )
   ```

4. **流水线通信重叠约束**：
   ```python
   # 错误：warmup/flush 重叠需要 overlap_p2p_comm 但不能用 batch_p2p_comm
   config = ModelParallelConfig(
       overlap_p2p_comm_warmup_flush=True,
       overlap_p2p_comm=False,  # ❌ 会抛出 ValueError
       batch_p2p_comm=True,     # ❌ 会抛出 ValueError
   )
   ```

## 常见配置示例

### 示例 1：基础 GPT 训练（单机 8 卡）

```python
config = ModelParallelConfig(
    tensor_model_parallel_size=8,  # 8 卡张量并行
    pipeline_model_parallel_size=1,
    sequence_parallel=True,        # 启用序列并行节省内存
    bf16=True,                     # 使用 BF16 混合精度
)
```

### 示例 2：大规模 GPT 训练（多节点）

```python
config = ModelParallelConfig(
    # 并行配置（假设 64 卡：8 节点 * 8 卡）
    tensor_model_parallel_size=8,          # TP=8
    pipeline_model_parallel_size=4,        # PP=4
    virtual_pipeline_model_parallel_size=2, # VPP=2（减少气泡）
    sequence_parallel=True,                 # 序列并行
    
    # 精度配置
    bf16=True,
    params_dtype=paddle.bfloat16,
    
    # 优化配置
    gradient_accumulation_fusion=True,      # 梯度累积融合
    tp_comm_overlap=True,                   # TP 通信重叠
    tp_comm_bulk_wgrad=True,
    tp_comm_bulk_dgrad=True,
    
    # 流水线配置
    batch_p2p_comm=True,
    deallocate_pipeline_outputs=True,       # 节省内存
)
```

### 示例 3：MoE 模型训练

```python
config = ModelParallelConfig(
    # 基础并行
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    sequence_parallel=True,  # EP+TP 时必需
    
    # MoE 专家并行
    expert_model_parallel_size=8,           # 8 个专家并行组
    expert_tensor_parallel_size=2,          # 专家内部 TP=2
    
    # 精度
    bf16=True,
)
```

### 示例 4：超长序列训练

```python
config = ModelParallelConfig(
    # 基础并行
    tensor_model_parallel_size=4,
    sequence_parallel=True,
    
    # 上下文并行处理长序列
    context_parallel_size=8,                # 将序列切分到 8 卡
    
    # 精度和优化
    bf16=True,
    gradient_accumulation_fusion=True,
)
```

### 示例 5：内存受限场景

```python
config = ModelParallelConfig(
    # 并行配置
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,
    sequence_parallel=True,
    
    # 内存优化
    deallocate_pipeline_outputs=True,
    cpu_offloading=True,                    # CPU offloading
    cpu_offloading_num_layers=8,            # 卸载 8 层
    
    # 梯度检查点
    num_microbatches_with_partial_activation_checkpoints=4,
    
    # 精度
    bf16=True,
)
```

### 示例 6：从检查点恢复训练

```python
config = ModelParallelConfig(
    # 并行配置
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    sequence_parallel=True,
    
    # 跳过初始化（从检查点加载）
    perform_initialization=False,
    
    # 精度
    bf16=True,
)
```

## 性能调优建议

### 1. 选择合适的并行策略

**小模型（< 10B 参数）**：
- 主要使用数据并行
- TP = 1-2（如果需要）
- PP = 1

**中等模型（10B-70B 参数）**：
- DP + TP 组合
- TP = 4-8
- PP = 1-2（如果单节点内存不足）
- 启用 sequence_parallel

**大模型（> 70B 参数）**：
- DP + TP + PP 组合
- TP = 8-16
- PP = 4-8
- VPP = 2-4（减少气泡）
- 启用 sequence_parallel
- 考虑 context_parallel（超长序列）

### 2. 通信优化

```python
config = ModelParallelConfig(
    # 启用张量并行通信重叠
    tp_comm_overlap=True,
    tp_comm_bulk_wgrad=True,
    tp_comm_bulk_dgrad=True,
    tp_comm_overlap_ag=True,
    tp_comm_overlap_rs=True,
    
    # 流水线通信批量化
    batch_p2p_comm=True,
    batch_p2p_sync=True,
)
```

### 3. 内存优化

```python
config = ModelParallelConfig(
    # 序列并行（减少激活内存）
    sequence_parallel=True,
    
    # 释放流水线输出
    deallocate_pipeline_outputs=True,
    
    # 梯度检查点
    num_microbatches_with_partial_activation_checkpoints=4,
    
    # CPU offloading（极端情况）
    cpu_offloading=True,
    cpu_offloading_num_layers=12,
)
```

### 4. 计算优化

```python
config = ModelParallelConfig(
    # 混合精度
    bf16=True,  # 推荐 Ampere+ GPU
    
    # 梯度累积融合
    gradient_accumulation_fusion=True,
    
    # 交叉熵融合
    cross_entropy_loss_fusion=True,
    cross_entropy_fusion_impl="native",
    
    # Flash Attention
    fa_version=3,  # 如果支持 FA3
)
```

## 故障排查

### 问题 1：OOM（内存不足）

**症状**：训练时 GPU 内存溢出

**解决方案**：
1. 增加模型并行度（TP 或 PP）
2. 启用 `sequence_parallel=True`
3. 设置 `deallocate_pipeline_outputs=True`
4. 减少 batch size 或增加梯度累积步数
5. 使用梯度检查点
6. 最后尝试 CPU offloading

### 问题 2：训练速度慢

**症状**：训练吞吐量低于预期

**解决方案**：
1. 启用通信重叠：`tp_comm_overlap=True`
2. 使用虚拟流水线并行减少气泡
3. 调整 `microbatch_group_size_per_vp_stage`
4. 启用 `batch_p2p_comm=True`
5. 检查 TP/PP 的配置是否合理

### 问题 3：Loss 不收敛或 NaN

**症状**：训练不稳定

**解决方案**：
1. 使用 BF16 而不是 FP16（更稳定）
2. 启用 `deterministic_mode=True` 进行调试
3. 检查学习率和梯度裁剪
4. 确保所有并行维度的梯度正确同步

### 问题 4：流水线并行死锁

**症状**：训练挂起

**解决方案**：
1. 检查 `overlap_p2p_comm` 和 `batch_p2p_comm` 的互斥约束
2. 确保 micro-batch 数量 >= PP size
3. 检查 `barrier_with_L1_time` 设置
4. 验证所有 rank 的代码路径一致

## 相关文件

- `src/paddlefleet/parallel_state.py` - 并行状态管理
- `src/paddlefleet/training/initialize.py` - 使用配置初始化训练
- `src/paddlefleet/transformer/transformer_config.py` - Transformer 配置（继承自 ModelParallelConfig）
- `src/paddlefleet/models/gpt/gpt_config.py` - GPT 配置（继承自 TransformerConfig）

## 参考文献

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
2. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)
3. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
4. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
5. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

## 总结

`ModelParallelConfig` 是 PaddleFleet 的核心配置类，包含了 70+ 个配置参数，涵盖：

- **6 种并行策略**：张量并行、流水线并行、虚拟流水线并行、序列并行、上下文并行、专家并行
- **多种优化技术**：通信重叠、梯度融合、CPU offloading
- **灵活的训练配置**：混合精度、梯度检查点、确定性模式
- **完善的验证机制**：自动调整和约束检查

理解和正确配置这些参数对于高效的大规模分布式训练至关重要。建议从简单配置开始，逐步增加优化选项，并根据实际训练情况进行调优。
