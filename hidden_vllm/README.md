# Origin vLLM - 基于继承的扩展实现

## 概述

Origin vLLM 是一个基于继承的 vLLM 扩展实现，通过继承现有的 vLLM 基础类来提供增强功能，同时保持完全的 API 兼容性。

## 特性

- **完全继承**: 通过继承原始 vLLM 类来扩展功能
- **API 兼容**: 与原始 vLLM API 完全兼容
- **灵活配置**: 支持动态配置和扩展
- **安全回退**: 当基础 vLLM 不可用时提供安全回退机制
- **易于扩展**: 简单的接口用于添加自定义功能

## 目录结构

```
origin_vllm/
├── __init__.py                 # 主入口文件
├── config.py                   # 继承配置系统
├── setup.py                    # 环境设置脚本
├── version.py                  # 版本信息
├── commit_id.py               # 提交 ID
├── engine/                     # 引擎模块
│   ├── __init__.py
│   ├── arg_utils.py           # 参数工具类
│   ├── llm_engine.py          # LLM 引擎
│   └── async_llm_engine.py    # 异步 LLM 引擎
├── entrypoints/               # 入口点模块
│   ├── __init__.py
│   └── llm.py                 # LLM 入口点
└── model_executor/            # 模型执行器模块
    ├── __init__.py
    └── models.py              # 模型注册表
```

## 快速开始

### 1. 环境设置

首先运行设置脚本来配置环境：

```bash
cd /home/hfd24/simpleRL-reason/origin_vllm
python setup.py
```

### 2. 基本使用

```python
import origin_vllm
from origin_vllm import LLM, SamplingParams

# 创建 LLM 实例（继承自原始 vLLM）
model = LLM(model="facebook/opt-125m")

# 创建采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 生成文本
outputs = model.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### 3. 使用示例脚本

```bash
python /home/hfd24/simpleRL-reason/example_origin_vllm.py
```

## 核心组件

### 1. 继承配置系统 (`config.py`)

提供了一个强大的配置系统来管理继承行为：

- `InheritanceConfig`: 管理继承配置的主类
- `safe_inherit()`: 安全地创建继承类，支持回退机制
- `get_base_class()`: 安全地导入基类
- `create_inherited_class()`: 动态创建继承类

### 2. 核心继承类

#### LLM 类 (`entrypoints/llm.py`)
- 继承自 `vllm.entrypoints.llm.LLM`
- 提供增强的文本生成功能
- 支持自定义扩展

#### LLMEngine 类 (`engine/llm_engine.py`)
- 继承自 `vllm.engine.llm_engine.LLMEngine`
- 增强的引擎功能
- 支持自定义请求处理

#### AsyncLLMEngine 类 (`engine/async_llm_engine.py`)
- 继承自 `vllm.engine.async_llm_engine.AsyncLLMEngine`
- 异步引擎功能增强
- 支持流式处理扩展

#### ModelRegistry 类 (`model_executor/models.py`)
- 继承自 `vllm.model_executor.models.ModelRegistry`
- 增强的模型注册功能
- 支持自定义模型类型

### 3. 参数处理 (`engine/arg_utils.py`)
- `EngineArgs`: 增强的引擎参数处理
- `AsyncEngineArgs`: 异步引擎参数处理
- 完全兼容原始 vLLM 参数系统

## 扩展指南

### 添加自定义功能

1. **扩展 LLM 类**:
```python
from origin_vllm.entrypoints.llm import LLM

class CustomLLM(LLM):
    def custom_generate(self, prompts, **kwargs):
        # 自定义生成逻辑
        results = super().generate(prompts, **kwargs)
        # 添加后处理
        return self.post_process(results)
    
    def post_process(self, results):
        # 自定义后处理逻辑
        return results
```

2. **使用配置系统**:
```python
from origin_vllm.config import get_config, safe_inherit

config = get_config()
config.add_custom_extension("my_feature", my_custom_function)

# 创建自定义继承类
MyClass = safe_inherit('vllm.some.module', 'SomeClass', 'MyClass', {
    'custom_method': my_custom_method
})
```

### 配置选项

```python
from origin_vllm.config import get_config

config = get_config()

# 设置基础 vLLM 路径
config.set_base_vllm_path("/path/to/vllm")

# 启用/禁用回退模式
config.enable_fallback_mode(True)

# 添加自定义扩展
config.add_custom_extension("logging", custom_logger)
```

## 与 our_vllm 的对比

| 特性 | our_vllm | origin_vllm |
|------|----------|-------------|
| 实现方式 | 直接修改 | 继承扩展 |
| 兼容性 | 定制化 | 完全兼容 |
| 维护性 | 需要手动同步 | 自动继承更新 |
| 扩展性 | 直接修改源码 | 通过继承扩展 |
| 安全性 | 修改原始代码 | 保留原始代码 |

## 优势

1. **保持原始代码完整性**: 不修改基础 vLLM 代码
2. **易于维护**: 基础功能更新时自动继承
3. **灵活扩展**: 可以轻松添加或移除功能
4. **向后兼容**: 完全兼容原始 vLLM API
5. **安全回退**: 在基础库不可用时提供安全回退

## 故障排除

### 常见问题

1. **导入错误**: 确保 vLLM 基础库已正确安装
2. **路径问题**: 运行 `setup.py` 配置正确的路径
3. **版本不兼容**: 检查基础 vLLM 版本是否为 0.5.4

### 调试

```python
from origin_vllm.config import get_config

config = get_config()
print(f"Base vLLM path: {config.base_vllm_path}")
print(f"Fallback mode: {config.fallback_mode}")
```

## 示例用例

### 1. 批量推理
```python
from origin_vllm import LLM, SamplingParams

model = LLM(model="facebook/opt-125m")
prompts = ["Hello", "How are you?", "What is AI?"]
sampling_params = SamplingParams(temperature=0.8)

outputs = model.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### 2. 异步推理
```python
import asyncio
from origin_vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def async_generate():
    engine_args = AsyncEngineArgs(model="facebook/opt-125m")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    sampling_params = SamplingParams(temperature=0.8)
    request_id = 0
    
    results = engine.generate("Hello world", sampling_params, request_id)
    async for output in results:
        print(output.outputs[0].text)

asyncio.run(async_generate())
```

## 许可证

本项目继承 vLLM 的许可证条款。

## 贡献

欢迎贡献代码！请确保：
1. 保持与原始 vLLM API 的兼容性
2. 添加适当的测试
3. 更新文档

## 支持

如有问题，请查看：
1. vLLM 官方文档
2. 本项目的示例代码
3. 配置和故障排除部分
