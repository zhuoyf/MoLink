# vLLM 0.29.0 升级与验证

上游基线为 `v0.29.0`，commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`。安装步骤见
[README](../README.md#quick-start)，补丁见
[`patch/`](../patch/README.md)。

## 当前动态 PP 版本

当前补丁来自 Runtime release 提交
[`00eef439c3`](https://github.com/zhuoyf/MoLink-runtime/commit/00eef439c33f1c4f5305120d49e3fc042062575a)，
包含本次 Admission Gate 校验异常修复及回归测试，
补丁对应实测 Git tree `f58512ea2b1091b7506fea75ebf357976e363744`。
运行时包含动态分层调度、Profiling、权重和 KV 在线迁移、Layer Stacking、
非连续 Attention、Dirty Bitmap、增量 Patch、安全 Commit 与 Admission Gate。
安装与启动入口见 [复现指南](REPRODUCE.md#动态-pp单机双-gpu)。

2026-11-15 对当前 release 重新启动并验证，在 2×RTX 3090 Ti、Llama-2-7B-Chat、
BF16、PP2/TP1、V2、eager、Triton Attention 上完成 17 项功能检查，使用 `max_model_len=512`、
`max_num_seqs=8`、`max_num_batched_tokens=128`、128 个 KV blocks：

- 静态推理与 `16/16 → 18/14 → 16/16` 双向在线迁移通过。
- 每个迁移方向运行 2 个并发请求、各生成 256 token；合计 1024 个迁移期间生成的 token
  与静态基线完全一致，两端 KV 摘要一致。两个方向分别记录到 11、17 个迁移期间推理步。
- KV `128 → 96 → 128` 缩扩容、容量预检、非法切分拒绝后的继续生成通过。
- Admission Gate 阻止新请求，租约到期后恢复；错误 owner 操作返回 409。
  复测时发现原 release 的校验异常经 RPC 后变成 500，现改用现有的
  `ReconfigurationValidationError` 保留错误分类，并增加 5 项回归。
- 流式输出、断连后继续生成、长提示词分块 prefill、资源监控及手动模式调度状态通过。
- 往返迁移后的实际分配显存恢复，停止测试服务后 GPU 显存恢复基线。
- 调度、环境变量、Worker RPC 和准入回归共 89 项通过；修复文件的 pre-commit 全部通过。
- 补丁应用到上游基线后，经 `git apply --cached --check` 和 `git write-tree`
  验证，Git tree 与实测源码一致。

环境、token、KV 校验与检查结果保存在
[`dynamic-pp-release-0.29.0.json`](../data/validation/dynamic-pp-release-0.29.0.json)。
本次服务直接使用容器的 64 MiB `/dev/shm`，设置
`NCCL_BUFFSIZE=1048576`、`NCCL_SHM_DISABLE=0`、`VLLM_MQ_MAX_CHUNK_BYTES_MB=1`。
详细脚本和日志保留在工作区 `/workspace/runs/release-00eef439c3-20261115/`；
回归命令见结果 JSON 的 `regression_commands`。

这组检查不包含跨机部署、CUDA graphs、匹配校准文件的自动调度、长时间性能或全量精度测量。
下文保留 2026-11-05 的链路升级验证记录。

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
