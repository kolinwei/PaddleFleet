# PaddleFleet 代码结构详解

本文档提供 PaddleFleet 项目的详细代码结构说明，包括主要类、函数、接口和代码组织方式。

## 目录

- [项目组织结构](#项目组织结构)
- [核心 API 参考](#核心-api-参考)
- [模块间依赖关系](#模块间依赖关系)
- [代码风格与规范](#代码风格与规范)
- [扩展开发指南](#扩展开发指南)

## 项目组织结构

### 源代码布局

```
PaddleFleet/
├── src/paddlefleet/          # 主包源代码
│   ├── __init__.py           # 包初始化，导出核心 API
│   ├── package_info.py       # 包信息（版本、作者等）
│   ├── spec_utils.py         # LayerSpec 规格工具
│   ├── timers.py             # 性能计时器
│   ├── utils.py              # 通用工具函数
│   ├── jit.py                # JIT 编译支持
│   ├── parallel_state.py     # 并行状态管理
│   ├── model_parallel_config.py  # 模型并行配置
│   ├── process_groups_config.py  # 进程组配置
│   ├── context_parallel_utils.py # 上下文并行工具
│   ├── config_logger.py      # 日志配置
│   ├── recompute_utils.py    # 重计算工具
│   ├── gpt_builders.py       # GPT 构建器
│   ├── packed_seq_params.py  # 打包序列参数
│   └── [子模块目录]/
├── tests/                    # 测试套件
│   └── single_card_tests/    # 单卡测试
├── ci/                       # 持续集成脚本
├── third_party/              # 第三方依赖
├── setup.py                  # 安装脚本
├── pyproject.toml            # 项目配置
├── backends.py               # 后端检测
└── build_utils.py            # 构建工具

代码统计：
- Python 文件：110+ 个
- CUDA 文件：14 个
- 代码行数：约 20,000+ 行（不含注释和空行）
```

## 核心 API 参考

### 1. 顶层 API (`paddlefleet`)

#### 1.1 初始化

```python
from paddlefleet.training import initialize_fleet

def initialize_fleet(
    args=None,
    training_args=None,
    model_parallel_config=None,
    extra_args_provider=None,
):
    """
    初始化 PaddleFleet 训练环境
    
    参数：
        args: 命令行参数
        training_args: 训练参数对象
        model_parallel_config: 模型并行配置
        extra_args_provider: 额外参数提供函数
        
    返回：
        初始化后的参数对象
    """
```

#### 1.2 并行状态 (`paddlefleet.parallel_state`)

```python
import paddlefleet.parallel_state as mpu

# 进程组管理
mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
)

# 获取并行信息
tensor_parallel_rank = mpu.get_tensor_model_parallel_rank()
tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()
pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()

# 获取通信组
tensor_parallel_group = mpu.get_tensor_model_parallel_group()
pipeline_parallel_group = mpu.get_pipeline_model_parallel_group()
data_parallel_group = mpu.get_data_parallel_group()
```

#### 1.3 层规格 (`LayerSpec`)

```python
from paddlefleet import LayerSpec

# 定义层规格
layer_spec = LayerSpec(
    module_class=TransformerLayer,
    config=transformer_config,
    sublayers=sublayer_specs,
)

# 使用层规格构建模型
layers = [layer_spec.build() for _ in range(num_layers)]
```

#### 1.4 计时器 (`Timers`)

```python
from paddlefleet import Timers

timers = Timers()
timers.start("forward")
# ... 前向传播代码
timers.stop("forward")
elapsed = timers.get("forward")
```

### 2. Transformer API (`paddlefleet.transformer`)

#### 2.1 核心类

```python
from paddlefleet.transformer import TransformerConfig, FleetLayer

# 配置 Transformer
config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    ffn_hidden_size=4096,
    num_attention_heads=16,
    # ... 更多配置选项
)

# 创建 Transformer 层
class MyTransformerLayer(FleetLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = SelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, hidden_states, attention_mask):
        # 实现前向传播
        pass
```

#### 2.2 注意力机制

```python
from paddlefleet.transformer.attention import CoreAttention
from paddlefleet.transformer.dot_product_attention import DotProductAttention

# 创建注意力层
attention = DotProductAttention(
    config=config,
    layer_number=1,
    attn_mask_type=AttnMaskType.causal,
)

# 前向传播
context, attention_probs = attention(
    query, key, value,
    attention_mask=attention_mask,
)
```

#### 2.3 MLP (前馈网络)

```python
from paddlefleet.transformer.mlp import MLP, MLPSubmodules

# 创建 MLP
mlp = MLP(
    config=config,
    submodules=MLPSubmodules(
        linear_fc1=ColumnParallelLinear(...),
        linear_fc2=RowParallelLinear(...),
    ),
)

# 前向传播
output = mlp(hidden_states)
```

#### 2.4 MoE 层

```python
from paddlefleet.transformer.moe import MoELayer, MoELayerParams

# 创建 MoE 层
moe_params = MoELayerParams(
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,
    eval_capacity_factor=2.0,
    expert_parallel_size=2,
)

moe_layer = MoELayer(
    config=config,
    params=moe_params,
    layer_number=1,
)

# 前向传播
output, aux_loss = moe_layer(hidden_states)
```

### 3. Models API (`paddlefleet.models`)

#### 3.1 GPT 模型

```python
from paddlefleet.models.gpt import GPTModel, GPTConfig

# 配置 GPT
gpt_config = GPTConfig(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    vocab_size=50257,
    max_position_embeddings=2048,
    # ... 更多配置
)

# 创建 GPT 模型
model = GPTModel(
    config=gpt_config,
    transformer_layer_spec=layer_spec,
    vocab_size=50257,
    max_sequence_length=2048,
)

# 前向传播
logits = model(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
```

#### 3.2 ViT 模型

```python
from paddlefleet.models.vision import CLIPViTModel

# 创建 ViT 模型
vit_model = CLIPViTModel(
    config=transformer_config,
    img_h=224,
    img_w=224,
    patch_size=16,
    num_classes=1000,
)

# 前向传播
output = vit_model(images)
```

#### 3.3 多模态模型

```python
from paddlefleet.models.multimodal import LLaVAModel

# 创建 LLaVA 模型
llava_model = LLaVAModel(
    language_transformer_config=language_config,
    vision_transformer_config=vision_config,
    language_transformer_layer_spec=language_spec,
    vision_transformer_layer_spec=vision_spec,
)

# 前向传播
logits = llava_model(
    images=images,
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
```

### 4. Tensor Parallel API (`paddlefleet.tensor_parallel`)

#### 4.1 并行线性层

```python
from paddlefleet.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# 列并行线性层（输出维度切分）
column_parallel = ColumnParallelLinear(
    input_size=1024,
    output_size=4096,
    bias=True,
    gather_output=False,  # 是否聚合输出
    init_method=init.normal_,
    config=config,
)

# 行并行线性层（输入维度切分）
row_parallel = RowParallelLinear(
    input_size=4096,
    output_size=1024,
    bias=True,
    input_is_parallel=True,  # 输入已经是切分的
    init_method=init.normal_,
    config=config,
)

# 词表并行嵌入
vocab_embedding = VocabParallelEmbedding(
    num_embeddings=50257,
    embedding_dim=1024,
    init_method=init.normal_,
    config=config,
)
```

#### 4.2 张量并行映射

```python
from paddlefleet.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)

# 复制到张量并行区域
x_parallel = copy_to_tensor_model_parallel_region(x)

# 从张量并行区域归约
x_reduced = reduce_from_tensor_model_parallel_region(x_parallel)

# 分散到张量并行区域
x_scattered = scatter_to_tensor_model_parallel_region(x)

# 从张量并行区域聚合
x_gathered = gather_from_tensor_model_parallel_region(x_scattered)
```

#### 4.3 并行交叉熵

```python
from paddlefleet.tensor_parallel import vocab_parallel_cross_entropy

# 计算词表并行的交叉熵损失
loss = vocab_parallel_cross_entropy(
    vocab_parallel_logits=logits,  # [batch, seq_len, vocab_size/tp]
    target=labels,                  # [batch, seq_len]
)
```

#### 4.4 随机数和检查点

```python
from paddlefleet.tensor_parallel import (
    checkpoint,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)

# 设置模型并行随机种子
model_parallel_cuda_manual_seed(seed=12345)

# 获取 CUDA 随机数跟踪器
rng_tracker = get_cuda_rng_tracker()

# 使用梯度检查点
def custom_forward(hidden_states):
    return layer(hidden_states)

output = checkpoint(custom_forward, hidden_states)
```

### 5. Pipeline Parallel API (`paddlefleet.pipeline_parallel`)

#### 5.1 定义流水线层

```python
from paddlefleet.pipeline_parallel import (
    LayerDesc,
    SharedLayerDesc,
    PipelineLayer,
)

# 定义层描述
layer_descs = [
    LayerDesc(EmbeddingLayer, config=config),
    *[LayerDesc(TransformerLayer, config=config) 
      for _ in range(num_layers)],
    LayerDesc(OutputLayer, config=config),
]

# 定义共享层（如输入和输出嵌入共享权重）
shared_layer = SharedLayerDesc(
    "embedding",
    EmbeddingLayer,
    forward_input_key="input_ids",
    shared_weight_attr="word_embeddings.weight",
)

# 创建流水线层
pipeline_model = PipelineLayer(
    layers=layer_descs,
    num_stages=4,  # 流水线阶段数
    loss_fn=loss_function,
)
```

#### 5.2 流水线并行调度

```python
from paddlefleet.pipeline_parallel import (
    PipelineParallel,
    PipelineParallelWithInterleave,
    PipelineParallelWithInterleaveFthenB,
)

# 标准流水线并行（1F1B）
pp = PipelineParallel(
    layers=pipeline_model,
    loss_fn=loss_function,
    topology=topology,
)

# 虚拟流水线并行（VPP）
vpp = PipelineParallelWithInterleave(
    layers=pipeline_model,
    loss_fn=loss_function,
    topology=topology,
    num_model_chunks=2,  # 虚拟阶段数
)

# F-then-B 调度
fthenb = PipelineParallelWithInterleaveFthenB(
    layers=pipeline_model,
    loss_fn=loss_function,
    topology=topology,
    num_model_chunks=2,
)

# 执行训练步
loss = pp.train_batch(
    data_iter=data_iterator,
    forward_step_func=forward_step,
)
```

#### 5.3 流水线调度配置

```python
from paddlefleet.pipeline_parallel import ScheduleNode, ScheduleChunk

# 定义调度节点
schedule = [
    ScheduleNode(
        type="forward",
        chunk_id=0,
        micro_batch_id=0,
    ),
    ScheduleNode(
        type="backward",
        chunk_id=0,
        micro_batch_id=0,
    ),
    # ... 更多调度节点
]

# 定义调度块
chunk = ScheduleChunk(
    chunk_id=0,
    forward_nodes=[...],
    backward_nodes=[...],
)
```

### 6. Fusions API (`paddlefleet.fusions`)

#### 6.1 融合归一化

```python
from paddlefleet.fusions import (
    fused_layer_norm,
    fused_rms_norm,
)

# 融合的 Layer Normalization
output = fused_layer_norm(
    input=hidden_states,
    normalized_shape=[hidden_size],
    weight=ln_weight,
    bias=ln_bias,
    eps=1e-5,
)

# 融合的 RMS Normalization
output = fused_rms_norm(
    input=hidden_states,
    normalized_shape=[hidden_size],
    weight=rms_weight,
    eps=1e-5,
)
```

#### 6.2 融合激活函数

```python
from paddlefleet.fusions import (
    fused_bias_gelu,
    fused_bias_swiglu,
    fused_bias_geglu,
)

# 融合的 Bias + GELU
output = fused_bias_gelu(input=x, bias=bias)

# 融合的 Bias + SwiGLU
output = fused_bias_swiglu(
    input=x,
    bias=bias,
    gate=gate,
)

# 融合的 Bias + GEGLU
output = fused_bias_geglu(
    input=x,
    bias=bias,
    gate=gate,
)
```

#### 6.3 融合 Softmax

```python
from paddlefleet.fusions import fused_softmax

# 融合的 Softmax（支持 causal mask）
attention_probs = fused_softmax(
    input=attention_scores,
    mask=attention_mask,
    scale=1.0 / math.sqrt(head_dim),
)
```

### 7. Custom Ops API (`paddlefleet.ops`)

#### 7.1 算子可用性检查

```python
from paddlefleet.ops import (
    is_deep_gemm_available,
    is_deep_ep_available,
    is_sonic_moe_available,
)

# 检查 Deep GEMM 是否可用
if is_deep_gemm_available():
    from paddlefleet.ops import deep_gemm
    # 使用 Deep GEMM

# 检查 Deep EP 是否可用
if is_deep_ep_available():
    from paddlefleet.ops import deep_ep
    # 使用 Deep EP

# 检查 SonicMoE 是否可用
if is_sonic_moe_available():
    from paddlefleet.ops import sonicmoe
    # 使用 SonicMoE
```

#### 7.2 自定义 CUDA 算子

```python
from paddlefleet.ops import (
    fuse_transpose_split_fp8_quant,
    tokens_unzip_gather,
    fuse_swiglu_scale,
    # ... 更多自定义算子
)

# 使用自定义算子（示例）
output = fuse_swiglu_scale(
    input=x,
    gate=g,
    scale=scale_factor,
)
```

### 8. Refined Recompute API (`paddlefleet.refined_recompute`)

```python
from paddlefleet.refined_recompute import (
    flash_attn_with_recompute,
)

# Flash Attention 与重计算
context = flash_attn_with_recompute(
    query=q,
    key=k,
    value=v,
    dropout_p=0.1,
    causal=True,
    return_attn_probs=False,
)
```

## 模块间依赖关系

### 依赖层次结构

```
Level 0 (基础工具):
├── utils.py
├── config_logger.py
├── package_info.py
└── backends.py

Level 1 (核心配置):
├── parallel_state.py
├── model_parallel_config.py
├── process_groups_config.py
└── spec_utils.py

Level 2 (基础算子):
├── ops/
├── fusions/
├── _extensions/
└── tensor_parallel/

Level 3 (核心组件):
├── transformer/
├── pipeline_parallel/
└── refined_recompute/

Level 4 (模型实现):
├── models/gpt/
├── models/vision/
├── models/multimodal/
└── models/common/

Level 5 (训练框架):
├── training/
├── distributed/
└── gpt_builders.py
```

### 模块间调用关系

```
training.initialize_fleet()
    ↓
parallel_state.initialize_model_parallel()
    ↓
models.GPTModel()
    ↓
├── transformer.TransformerLayer()
│   ├── transformer.attention.Attention()
│   │   └── tensor_parallel.ColumnParallelLinear()
│   └── transformer.mlp.MLP()
│       ├── tensor_parallel.ColumnParallelLinear()
│       └── tensor_parallel.RowParallelLinear()
└── pipeline_parallel.PipelineLayer()
    └── pipeline_parallel.PipelineParallel()
```

## 代码风格与规范

### 1. 代码格式

PaddleFleet 使用 **Ruff** 进行代码格式化和检查：

```bash
# 格式化代码
ruff format .

# 检查代码
ruff check .

# 自动修复
ruff check --fix .
```

**配置**（在 `pyproject.toml` 中）：
- 行长度：80 字符
- 目标版本：Python 3.10+
- 使用 `isort` 进行导入排序
- 遵循 PEP 8 标准

### 2. 导入顺序

```python
# 1. 标准库导入
import os
import sys
from typing import Optional

# 2. 第三方库导入
import paddle
import numpy as np

# 3. PaddleFleet 内部导入
from paddlefleet import parallel_state as mpu
from paddlefleet.transformer import TransformerConfig
```

### 3. 类型注解

PaddleFleet 鼓励使用类型注解：

```python
from typing import Optional, Tuple, List

def forward(
    self,
    hidden_states: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """
    前向传播
    
    参数：
        hidden_states: 输入张量，形状 [batch, seq_len, hidden_size]
        attention_mask: 注意力掩码，形状 [batch, 1, seq_len, seq_len]
        
    返回：
        output: 输出张量
        attention_probs: 注意力概率（可选）
    """
    pass
```

### 4. 文档字符串

使用 Google 风格的文档字符串：

```python
class TransformerLayer(FleetLayer):
    """
    Transformer 层实现
    
    包含自注意力和前馈网络，支持 Pre-LN 和 Post-LN。
    
    参数：
        config: Transformer 配置对象
        layer_number: 层编号（从 1 开始）
        self_attn_mask_type: 注意力掩码类型
        
    属性：
        attention: 自注意力模块
        mlp: 前馈网络模块
        input_layernorm: 输入层归一化
        post_attention_layernorm: 注意力后归一化
        
    示例：
        >>> config = TransformerConfig(num_layers=24, hidden_size=1024)
        >>> layer = TransformerLayer(config, layer_number=1)
        >>> output = layer(hidden_states, attention_mask)
    """
    pass
```

### 5. 许可证头

所有源文件必须包含许可证头：

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## 扩展开发指南

### 1. 添加新的 Transformer 层

```python
# 1. 继承 FleetLayer
from paddlefleet.transformer import FleetLayer

class CustomTransformerLayer(FleetLayer):
    def __init__(self, config, layer_number):
        super().__init__(config=config)
        self.layer_number = layer_number
        
        # 添加自定义组件
        self.custom_attention = CustomAttention(config)
        self.custom_mlp = CustomMLP(config)
        
    def forward(self, hidden_states, attention_mask=None):
        # 实现前向传播
        attention_output = self.custom_attention(
            hidden_states, attention_mask
        )
        output = self.custom_mlp(attention_output)
        return output

# 2. 创建层规格
from paddlefleet import LayerSpec

custom_layer_spec = LayerSpec(
    module_class=CustomTransformerLayer,
    config=config,
)

# 3. 在模型中使用
model = GPTModel(
    config=gpt_config,
    transformer_layer_spec=custom_layer_spec,
)
```

### 2. 添加新的自定义算子

#### 2.1 Python 包装器

```python
# src/paddlefleet/_extensions/my_custom_op.py

import paddle

def my_custom_op(input: paddle.Tensor) -> paddle.Tensor:
    """
    自定义算子的 Python 接口
    
    参数：
        input: 输入张量
        
    返回：
        输出张量
    """
    # 调用 C++/CUDA 实现
    return paddle.incubate.my_custom_op(input)
```

#### 2.2 CUDA 实现

```cuda
// src/paddlefleet/_extensions/my_custom_op.cu

#include <paddle/extension.h>

__global__ void my_custom_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;  // 示例操作
    }
}

paddle::Tensor MyCustomOpCUDA(const paddle::Tensor& input) {
    auto output = paddle::empty_like(input);
    
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    my_custom_kernel<<<blocks, threads>>>(
        input.data<float>(),
        output.data<float>(),
        size
    );
    
    return output;
}

PADDLE_MODULE_INIT {
    paddle::RegisterCustomOp(
        "my_custom_op",
        &MyCustomOpCUDA,
        "My custom operation"
    );
}
```

#### 2.3 在 setup.py 中注册

```python
# setup.py

ext_module = CUDAExtension(
    sources=[
        # ... 其他源文件
        "./src/paddlefleet/_extensions/my_custom_op.cu",
    ],
    # ...
)
```

### 3. 添加新的融合算子

```python
# src/paddlefleet/fusions/fused_my_activation.py

import paddle

def fused_my_activation(
    input: paddle.Tensor,
    bias: paddle.Tensor,
) -> paddle.Tensor:
    """
    融合的自定义激活函数
    
    将 bias 加法和激活函数融合到一个算子中
    
    参数：
        input: 输入张量
        bias: 偏置张量
        
    返回：
        激活后的张量
    """
    # 方法 1: 使用 Paddle 内置融合算子
    return paddle.incubate.fused_bias_act(
        input, bias, act="my_activation"
    )
    
    # 方法 2: 使用自定义 CUDA 算子
    # from paddlefleet.ops import fused_my_activation_cuda
    # return fused_my_activation_cuda(input, bias)
```

### 4. 添加新的并行策略

```python
# src/paddlefleet/my_parallel/my_parallel_strategy.py

import paddle
from paddlefleet import parallel_state as mpu

class MyParallelStrategy:
    """
    自定义并行策略
    """
    
    def __init__(self, config):
        self.config = config
        self.parallel_size = config.my_parallel_size
        
    def setup_parallel_groups(self):
        """设置并行通信组"""
        # 创建自定义的进程组
        pass
        
    def parallelize_layer(self, layer):
        """
        并行化一个层
        
        参数：
            layer: 要并行化的层
            
        返回：
            并行化后的层
        """
        # 实现层的并行化逻辑
        pass
        
    def forward_with_parallel(self, layer, input):
        """
        执行并行化的前向传播
        
        参数：
            layer: 并行化的层
            input: 输入张量
            
        返回：
            输出张量
        """
        # 实现前向传播逻辑
        pass
```

### 5. 扩展配置系统

```python
# src/paddlefleet/training/my_config.py

from dataclasses import dataclass
from paddlefleet.transformer import TransformerConfig

@dataclass
class MyCustomConfig(TransformerConfig):
    """
    自定义配置类
    
    扩展基础的 TransformerConfig
    """
    
    # 添加新的配置字段
    use_custom_attention: bool = False
    custom_attention_type: str = "flash"
    custom_mlp_type: str = "gated"
    
    # 添加验证逻辑
    def __post_init__(self):
        super().__post_init__()
        
        if self.use_custom_attention:
            assert self.custom_attention_type in ["flash", "block"]
```

## 测试指南

### 1. 运行测试

```bash
# 运行所有单卡测试
pytest tests/single_card_tests/

# 运行特定测试
pytest tests/single_card_tests/model/test_gpt_model_dense.py

# 运行带覆盖率的测试
pytest --cov=paddlefleet tests/
```

### 2. 编写测试

```python
# tests/single_card_tests/test_my_feature.py

import unittest
import paddle
from paddlefleet import MyFeature

class TestMyFeature(unittest.TestCase):
    """测试自定义功能"""
    
    def setUp(self):
        """测试前设置"""
        paddle.set_device("gpu:0")
        
    def test_basic_functionality(self):
        """测试基本功能"""
        feature = MyFeature(config={})
        output = feature.process(input_data)
        self.assertIsNotNone(output)
        
    def test_edge_cases(self):
        """测试边界情况"""
        feature = MyFeature(config={})
        
        # 测试空输入
        with self.assertRaises(ValueError):
            feature.process(None)
            
    def tearDown(self):
        """测试后清理"""
        paddle.device.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
```

### 3. 测试组织

```
tests/
├── single_card_tests/       # 单卡测试
│   ├── model/              # 模型测试
│   │   ├── test_gpt_model_dense.py
│   │   ├── test_gpt_model_moe.py
│   │   └── test_llava_model.py
│   ├── transformer/        # Transformer 测试
│   │   ├── test_layer.py
│   │   ├── test_mlp.py
│   │   └── test_router.py
│   ├── custom_ops/         # 自定义算子测试
│   │   ├── test_fuse_swiglu_scale.py
│   │   └── deep_gemm/
│   └── test_*.py           # 其他单元测试
└── multi_card_tests/       # 多卡测试（如果有）
```

## 构建与部署

### 1. 本地开发安装

```bash
# 克隆仓库
git clone https://github.com/PaddlePaddle/PaddleFleet.git
cd PaddleFleet

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 构建 wheel 包

```bash
# 使用 uv
uv build

# 或使用 pip
python setup.py bdist_wheel
```

### 3. 后端特定构建

PaddleFleet 支持多种后端：

```python
# backends.py 检测当前环境

if paddle.is_compiled_with_cuda():
    IS_NVIDIA = True
    IS_XPU = False
elif paddle.is_compiled_with_xpu():
    IS_NVIDIA = False
    IS_XPU = True
else:
    # 不支持的后端
    pass
```

**NVIDIA GPU 后端**：
- 需要 CUDA 工具链
- 编译自定义 CUDA 算子
- 支持 Triton、Deep GEMM、SonicMoE 等

**昆仑芯 XPU 后端**：
- 使用 PaddlePaddle XPU 版本
- 不编译 CUDA 算子
- 使用 PaddlePaddle 原生算子

## 性能优化建议

### 1. 内存优化

- 使用梯度检查点（gradient checkpointing）
- 启用混合精度训练（FP16/BF16）
- 使用 FP8 量化（Hopper GPU）
- 调整 micro batch size

### 2. 计算优化

- 使用融合算子
- 启用 Flash Attention
- 使用编译优化（JIT）
- 调整张量并行度

### 3. 通信优化

- 使用梯度累积
- 启用通信重叠
- 使用 ZeRO 优化器
- 调整流水线并行调度

### 4. 配置示例

```python
config = TransformerConfig(
    # 基本配置
    hidden_size=4096,
    num_attention_heads=32,
    
    # 内存优化
    recompute_granularity="full",
    recompute_method="block",
    recompute_num_layers=1,
    
    # 计算优化
    use_flash_attn=True,
    use_fused_softmax=True,
    use_fused_rmsnorm=True,
    bf16=True,
    
    # 并行优化
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    sequence_parallel=True,
)
```

## 常见问题

### 1. 如何调试并行训练？

```python
# 设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用环境变量
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PADDLE_LOG_LEVEL"] = "INFO"

# 打印并行信息
import paddlefleet.parallel_state as mpu
print(f"TP rank: {mpu.get_tensor_model_parallel_rank()}")
print(f"PP rank: {mpu.get_pipeline_model_parallel_rank()}")
print(f"DP rank: {mpu.get_data_parallel_rank()}")
```

### 2. 如何处理 OOM（内存不足）？

- 减少 batch size
- 增加梯度累积步数
- 启用梯度检查点
- 增加模型并行度
- 使用 CPU offload（如果支持）

### 3. 如何选择并行策略？

- **小模型**：主要使用数据并行
- **中等模型**：数据并行 + 张量并行
- **大模型**：数据并行 + 张量并行 + 流水线并行
- **超大模型**：上述组合 + ZeRO + CPU offload

## 参考资源

### 官方文档

- PaddlePaddle 官方文档: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html
- PaddleFleet GitHub: https://github.com/PaddlePaddle/PaddleFleet

### 相关项目

- Megatron-LM: NVIDIA 的大规模语言模型训练框架
- DeepSpeed: Microsoft 的深度学习优化库
- FSDP: PyTorch 的全切片数据并行

### 学术论文

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

## 总结

PaddleFleet 提供了完整的大规模分布式训练解决方案。本文档涵盖了：

- 详细的模块和 API 参考
- 代码组织和风格规范
- 扩展开发指南
- 测试和构建流程
- 性能优化建议

对于更具体的使用示例和最佳实践，请参考项目中的测试代码和示例脚本。
