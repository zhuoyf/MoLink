# vLLM 0.29.0 升级与验证

上游基线为 `v0.29.0`，commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`。安装步骤见
[README](../README.md#quick-start)，补丁见
[`patch/`](../patch/README.md)。

## 适配内容

- 保留新调度器的输入预算、Mamba 对齐、KV cache 和推测解码字段，移植 TP-C 调度及反馈。
- 适配异步发送句柄：NCCL 的首个句柄是元数据，MoLink 链路返回的是载荷句柄；后者必须保留并在下一步等待。
- 0.29.0 已默认使用 Model Runner V2，并省去 V2 的 cached token 列表；保留 V1 的 token-tail 开关。
- 集群启动器使用 `vllm serve`、BF16，并通过
  `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=HETVLLM_` 向 Ray worker 传递链路参数。
- 预编译 wheel 固定为 0.29.0，与源码版本对应。

## 验证环境

2026-11-05，本机 2× RTX 3090 Ti，模型为 Llama-2-7B-Chat，BF16，
TP=1、PP=2、分区 `16,16`，`max_model_len=4096`、
`max_num_seqs=8`、`max_num_batched_tokens=1024`，eager 模式。

| 组件 | 版本 |
|---|---|
| Python | 3.12.16 |
| vLLM | 0.29.0+molink.cu129（可编辑源码安装） |
| PyTorch | 2.13.0+cu129 |
| Transformers | 5.22.1 |
| Ray | 2.60.0 |
| pyzmq | 27.3.0 |

本机容器的 `/dev/shm` 为 64 MiB。验证时使用 `proot` 将该路径映射到
`/workspace/molink-runtime/shm`，满足 vLLM 进程队列和 NCCL 内部共享内存池的空间需求。
这属于本机 IPC 的运行环境配置；阶段间载荷仍按配置走 NCCL 或 MoLink ZMQ。
正式容器应配置足够的共享内存，例如 `--shm-size=2g`。
本次运行用于正确性验证，不能用作性能基准。

## 回归测试

以下测试共 **26 项通过**，覆盖预测调度、token-tail、发送句柄、
异步 PP 调度、抢占、前缀缓存和请求终止：

```bash
.venv/bin/python -m pytest -q \
  tests/v1/core/test_scheduler.py::test_het_chunk_budget_preserves_decode \
  tests/v1/core/test_scheduler.py::test_cached_request_data_resumed_all_token_ids_mrv1_only \
  tests/v1/core/test_scheduler.py::test_schedule \
  tests/v1/core/test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders \
  tests/v1/core/test_async_scheduler.py \
  tests/v1/worker/test_gpu_worker.py::test_execute_model_waits_previous_pp_send_before_forward
```

## GPU 验证

以下四组使用 multiprocessing 执行器，Model Runner V2、异步调度均开启，
`batch_queue_size=3`，全部通过上述请求检查：

| 配置 | 链路与调度 |
|---|---|
| native | 原生 NCCL PP2 |
| bf16 | MoLink ZMQ，BF16 边界张量 |
| int4 | MoLink ZMQ，int4 边界编码、残差合并、token-tail |
| int4-pred | int4 配置加 TP-C 预测调度 |

BF16 链路的 3 组顺序输出、共 **96 个 token** 与原生 NCCL PP2 完全一致。
含长提示词的并发检查、流式输出和断连后的继续生成均通过。
各 MoLink 组均记录到链路实际发送的消息和字节。
结果及输出 token 见 [`data/validation/vllm-0.29.0.json`](../data/validation/vllm-0.29.0.json)。

运行时提交：[`311fc2b`](https://github.com/zhuoyf/MoLink-runtime/commit/311fc2b4d464d3b5bd872ffee7b65543af8e9fe6)。

验证脚本位于当前工作区 `/workspace/scripts/validate-029.py`，日志与响应保存在
`/workspace/runs/vllm-0.29.0/`：

```bash
cd /workspace/molink-runtime/vllm
proot -b /workspace/molink-runtime/shm:/dev/shm \
  .venv/bin/python /workspace/scripts/validate-029.py
```

每组执行 3 个顺序请求、4 个并发请求（其中一个约 1800 token，触发分块 prefill）、
流式输出，以及断连后的继续生成。顺序和并发请求各生成 32 token，
流式请求生成 16 token，断连后的请求生成 8 token。

历史性能及精度数字仍以 [EVALUATION.md](EVALUATION.md) 中的 0.23.0 测量为准。
本次验证范围是单机双卡 PP2；跨机 WAN、TP2×PP2、PP4、CUDA graphs 和全量精度评测需另行测量。
