# Local LLM Benchmarks — cachy-bbr

Benchmarks, verified server configs and long-context test scripts for the
local LLM stack on **cachy-bbr** (Benjamin's desktop). Modeled after
[kr4ckhe4d/local-llm-benchmarks](https://github.com/kr4ckhe4d/local-llm-benchmarks),
but for this box's NVIDIA hardware.

## Hardware (verified 2026-08-16)

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 8700G (8C/16T) |
| GPU 0 | NVIDIA RTX PRO 4500 Blackwell, 32 GB (32,623 MiB) |
| GPU 1 | NVIDIA GeForce RTX 5070 Ti, 16 GB (16,303 MiB) |
| VRAM total | ~48 GB |
| RAM | ~30 GB |
| Backends | llama.cpp CUDA builds (LM Studio 2.16.0 / 2.24.0 / 2.28.2, plus own build in `locallama/`), Ollama :11434 |
| Inference port | llama-server on `:8080` (0.0.0.0) |

`--split-mode layer` spreads a model across both GPUs; `--gpu-layers 99` +
`--fit on` for context fitting.

## Models on disk

| Model | Quant | Size | Type |
|---|---|---|---|
| Qwen3.6-35B-A3B | UD-Q4_K_XL | 22.9 GB | MoE 35B/~3B active, hybrid attention |
| Nemotron 3.5 Lightning 30B-A3B | UD-Q4_K_XL | 25.5 GB | MoE 30B/~3B active |
| Muse Glimmer 30B | UD-Q6_K_XL | 26.3 GB | Dense 30B |

## Headline findings (measured in earlier benchmark sessions on this box)

- **MoE beats dense for speed.** A 35B-A3B MoE hits ~94 tok/s on ONE 32 GB
  GPU and ~130 tok/s on both (layer split); a dense 27B stays ~30-46 tok/s.
  Architecture matters more than nominal size for agent/speed workloads.
- **MTP speculative decoding is real: +62%.** Dense Qwen3.6-27B Q8_0 went
  25.6 → 41.6 tok/s with `--spec-type draft-mtp --spec-draft-n-max 2` (the
  MTP draft lives inside the same GGUF — no separate draft file).
- **VRAM fit is a memory effect, not quality.** Dense 27B Q8_0 (~30 GB
  weights) maxes out at ~128k context on 48 GB — 204,768 fails with
  `cudaMalloc: out of memory`. The Q4 of the same model fits and runs 10×+
  faster. Drop hard-set `--gpu-layers` so `--fit on` can tune context/layers.
- **MoE long context is cheap.** 35B-A3B Q6 (~26 GB) runs 204,800 ctx × 4
  slots (kv-unified) comfortably on this box.
- **New architectures need new builds.** Nemotron-3.5 loads in LM Studio's
  bundled llama.cpp (2.28.2) but fails in Ollama's older bundled build
  (tensor mismatch — old llama.cpp, not a corrupt GGUF).
- **K-quants (Q4_K_XL / Q6_K_XL) are the sweet spot here** — all models on
  disk use Unsloth-style K-quants; nothing requires f16/Q8 for quality work.

## Current live server (2026-08-16)

Qwen3.6-35B-A3B-UD-Q4_K_XL via LM Studio bundled llama-server 2.28.2:

```
llama-server --model locallama/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 --temp 0.6 --top-p 0.95 --min-p 0.01 \
  --ctx-size 131072 --gpu-layers 99 --alias qwen3.6-35b
```

Resident VRAM: 17,760 MiB (GPU 0) + 8,202 MiB (GPU 1).

## Planned (scripts to add)

- [ ] `switch-model.sh` — stop/start llama-server per model/context preset
- [ ] `models-preset.ini` — Open WebUI router presets for all models
- [ ] `needle-test.py` — long-context recall decay (needle-in-haystack)
- [ ] `semantic-recall-test.py` — reuse-vs-reimplement semantic recall
- [ ] `coder-prompt.sh` — structured coding prompt builder w/ context budget
- [ ] Fresh `llama-bench` numbers per model/quant/context for the README