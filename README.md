<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink v2: Heterogeneous Pipeline Serving over Constrained Networks

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![vLLM 0.29.0](https://img.shields.io/badge/vLLM-0.29.0-green.svg)](https://github.com/vllm-project/vllm/tree/v0.29.0)

**MoLink** serves one large LLM across machines that are far apart or poorly connected, and across GPUs of different classes.
**MoLink v2** is a rewrite of that idea as a small, default-off patch on top of upstream vLLM v0.29.0,
instead of the gRPC-based engine fork of v1. It targets one specific problem: with pipeline parallelism over a
slow link (100 Mbps – 1 Gbps), the activation transfer between stages dominates every step, and the GPUs sit idle.

This branch also includes **dynamic PP**: GPU-aware profiling and partition selection,
online weight/KV migration, non-contiguous attention, layer stacking, and migration admission control.
The patch tracks Runtime commit [`0358302773`](https://github.com/zhuoyf/MoLink-runtime/commit/03583027739f8446dbd2007ec5d55f9f75dfdc6c).
The dynamic path supports local Llama, PP2/TP1, BF16, V2, eager execution and Triton Attention.

## What v2 changes

| Technique | What it does |
|---|---|
| **`pp_link`** | Cross-machine PP transport over ZMQ instead of NCCL `isend/irecv`: 32 KiB framing, two priority classes (decode can overtake in-flight prefill bytes), per-direction ack-window backpressure, fully asynchronous send/receive. |
| **int4 boundary encoding** | Stage-boundary activations are group-quantized to int4 (int8 also available). |
| **Residual merging** | Only `s = h + r` crosses the link; the next stage recovers `RMSNorm(s)` exactly with a fused add+RMSNorm. Together with int4, cross-machine bytes drop to 0.14× of stock. |
| **Model Runner V2** | Upstream 0.29.0 enables V2 by default; `--best` selects it explicitly. |
| **Control-plane trimming** (`--tok-tail`) | Upstream 0.29.0 already omits cached token lists for V2; the switch retains tail-only payloads for V1. |
| **Predictive chunk scheduler** (optional, `--best-tpc`) | Fits step latency online and sizes prefill chunks to protect decode latency. |

Link optimizations use `HETVLLM_*`; dynamic PP uses `VLLM_PP_RECONFIGURE_*`.
The dynamic launcher `scripts/serve.sh` sets the required runtime options. Automatic partition scheduling additionally requires a matching profile.

## Results (headline)

The results below were measured on vLLM **0.23.0**; the 0.29.0 port has separate
[validation results](docs/UPGRADE-0.29.0.md).

Testbed: A = 2×RTX 4090, B = 2×RTX 3090, link throttled with `tc`. Same machines, same split, same time window as the stock-vLLM baseline.

| Setup | Result |
|---|---|
| Qwen3-30B-A3B, TP2×PP2, 40 paired tests, 100 Mbps – 1 Gbps | **1.91×** geometric-mean speedup vs stock vLLM (1.54× vs stock with Model Runner V2 forced) |
| Long-prompt prefill, 100 Mbps, 48 × 8K prompts | 52.1 s vs 327.8 s (**6.29×**) |
| Overload at 250 Mbps (300 s of sustained arrivals) | 153/155 requests meet all latency targets vs 0/155; median TTFT 1.0 s vs 49.2 s |
| Qwen3-32B (dense), TP2×PP2 | 1.25× (1.67× at 100 Mbps) |
| Qwen3-30B-A3B, PP4 (4090, 4090 \| 3090, 3090) | 2.53× vs stock PP4 (1.90× vs stock V2) |
| Qwen3-8B, PP2 (1×4090 + 1×3090) | 1.97×; bubble rate 41%/39% vs 89%/81% |

The advantage grows as the network gets slower. Boundary int4 quantization costs some accuracy; see
[EVALUATION §8](docs/EVALUATION.md) for measured numbers and [§10](docs/EVALUATION.md) for limitations.
All numbers are from one cluster and one snapshot of the code; they do not transfer to other hardware without re-measuring.

## Quick start

Install the patch and pinned Python 3.12 / CUDA 12.9 dependencies:

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 98dff2a81d747d1dba01a47f939f48c3526d4206          # v0.29.0
git apply --check /path/to/MoLink/patch/hetvllm-vllm-v0.29.0.patch
git apply /path/to/MoLink/patch/hetvllm-vllm-v0.29.0.patch
uv venv --python 3.12 && . .venv/bin/activate
uv pip install --python .venv/bin/python --no-deps -r requirements/runtime-cu129.txt
WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
VLLM_VERSION_OVERRIDE=0.29.0+molink.cu129 VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
    uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
uv pip check --python .venv/bin/python
```

For dynamic PP on two local GPUs, run from the patched vLLM directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve.sh /path/to/llama \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 128 \
  --num-gpu-blocks-override 128 --gpu-memory-utilization 0.8
```

The server listens on `127.0.0.1:8000`. It starts with a fixed partition and supports online changes:

```bash
.venv/bin/python -m vllm.distributed.pp_reconfigure monitor --samples 1
.venv/bin/python -m vllm.distributed.pp_reconfigure reconfigure --boundary 18
```

For a 32-layer Llama, boundary 18 selects an `18/14` partition. See the
[dynamic PP guide](docs/REPRODUCE.md#动态-pp单机双-gpu) for profiling, automatic scheduling,
container shared-memory settings and runtime constraints. TEE support is an extension interface only.

The two-machine link-optimization kit uses Ray and its own cluster configuration.
Install the patched runtime on both machines, then follow
[the cluster instructions](docs/REPRODUCE.md#2-配置). Machine B requires key-based SSH access.

## Repository layout

```
README.md              this file
docs/EVALUATION.md     full report: method, techniques, results, ablation, accuracy, limitations
docs/REPRODUCE.md      dynamic PP setup and two-machine kit instructions
patch/                 patch against vLLM v0.29.0 + per-file notes and env-var reference
kit/                   cluster launcher, benchmark/replay, ablation, bubble-rate and accuracy tools
data/                  raw replay JSON and summary CSVs behind every number and figure
figs/                  figures used in the report
make_figs.py           regenerates figs/ and data/*.csv from data/
resources/images/      logo
```

## Relationship to MoLink v1

v1 (gRPC pipeline between nodes, vLLM 0.19.0+, `molink/` and `molinkv1/` packages) is kept on the `main` branch and is
not part of this branch. v2 does not reuse its code; it keeps the goal (LLM serving across distributed, heterogeneous,
consumer-grade resources) and replaces the mechanism.

## License

Apache 2.0, see [LICENSE](LICENSE). The patch applies to vLLM, which is also Apache 2.0.

## Citation

If you find MoLink useful for your research or projects, please cite our paper:

Lewei Jin, Kui Zhang, Yongqi Chen, Zhuoyifan, Renjie Li, Yi Gao, Bowei Yang, Zhengong Cai, and Wei Dong. Distributed LLM Serving on Consumer-Grade GPUs by Reconciling Computation and Communication. In *Findings of the Association for Computational Linguistics: EMNLP 2025*, pages 17633–17642, Suzhou, China, November 2025. Association for Computational Linguistics.

```bibtex
@inproceedings{jin-etal-2025-distributed,
  title = {Distributed {LLM} Serving on Consumer-Grade {GPU}s by Reconciling Computation and Communication},
  author = {Jin, Lewei and Zhang, Kui and Chen, Yongqi and Zhuoyifan and Li, Renjie and Gao, Yi and Yang, Bowei and Cai, Zhengong and Dong, Wei},
  editor = {Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  month = nov,
  year = {2025},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  pages = {17633--17642},
  doi = {10.18653/v1/2025.findings-emnlp.957/},
  url = {https://aclanthology.org/2025.findings-emnlp.957/},
  isbn = {979-8-89176-335-7}
}
```
