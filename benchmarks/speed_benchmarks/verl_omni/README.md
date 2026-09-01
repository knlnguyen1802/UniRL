# vs verl-omni

[verl-omni](https://github.com/verl-project/verl-omni) is the closest peer framework
(diffusion / unified / omni RL on vLLM-Omni rollout + FSDP2/VeOmni training). Its
published reference number (~25% end-to-end over diffusers-based flow_grpo on
Qwen-Image FlowGRPO) comes without a public setup spec, so any comparison we publish
pins and discloses both sides per the protocol in [`../README.md`](../README.md).

The [`upstream/`](upstream/) **git submodule pins the exact verl-omni commit every
number below was measured against** (no code copied; `git submodule update --init
benchmarks/speed_benchmarks/verl_omni/upstream` to fetch). Bump the pin only together
with a re-run of the pair.

Two runnable aligned pairs live here: **SD3.5 + FlowGRPO + PickScore** (measured)
and **Qwen-Image + FlowGRPO + OCR** (pinned from verl-omni's
`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`; fill the
measured table after a run).

## Hands-on: the aligned SD3.5 + FlowGRPO pair

One aligned workload, three runnable commands. Both sides: 48 prompts/step ×
16 samples/prompt @384², 10 denoise steps, SDE noise on 3 of the first 5 steps,
2 mini-batch updates/step (micro 8/GPU), LoRA r32/α64 on the same 8 attention
projections, lr 1e-4 / wd 1e-4 / clip 1e-5, no KL/ref, the same
`yuvalkirstain/PickScore_v1` reward colocated with training, the same
`datasets/pickscore` prompts, 1×8 GPUs, val/save off.

```bash
# 0. one-time: fetch the pinned upstream + build both sides' prompt data
git submodule update --init benchmarks/speed_benchmarks/verl_omni/upstream
python benchmarks/speed_benchmarks/verl_omni/make_pickscore_parquet.py   # → ~/data/pickscore_sd3

# 1. UniRL side (UniRL env, repo root; console log → unirl.log)
SD35=stabilityai/stable-diffusion-3.5-medium STEPS=25 \
  bash benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh 2>&1 | tee unirl.log
python benchmarks/speed_benchmarks/parse_perf.py unirl.log --samples-per-step 768 --gpus 8

# 2. verl-omni side (verl-omni env per upstream/docs/start/install.md, same GPUs, one at a time)
SD35=stabilityai/stable-diffusion-3.5-medium STEPS=25 ATTN=sdpa \
  bash benchmarks/speed_benchmarks/verl_omni/run_verlomni_sd35_aligned.sh 2>&1 | tee verlomni.log
python benchmarks/speed_benchmarks/verl_omni/parse_verl_timing.py verlomni.log --samples-per-step 768 --gpus 8
```

`ATTN=sdpa` is the backend-aligned row (UniRL's diffusers/vLLM-Omni stack runs
SDPA-class kernels); `ATTN=fa3` runs verl-omni's own default attention (FA3 hub
kernel actor + FLASH_ATTN rollout) as its best-config row.

## Measured (2026-07-15, 1×8 H20 96GB, 25 steps, first 2 dropped)

| side | attention | median s/step | p90 | samples/GPU-h | rel. |
|---|---|---|---|---|---|
| **UniRL** (`sd3_vllmomni.yaml`, overrides above) | SDPA | **45.0** | 45.0 | **7680.0** | **1.00×** |
| verl-omni @ [`01c87ee`](https://github.com/verl-project/verl-omni/commit/01c87ee595874c313f9f296525fb5b4389678451) | SDPA-aligned | 111.4 | 111.8 | 3102.8 | 0.40× |
| verl-omni @ `01c87ee` | FA3 (its default) | 113.6 | 115.3 | 3042.2 | 0.40× |

Work-parity cross-check: after 25 steps all three runs sit at the same PickScore
(0.8101 / 0.8217 / 0.8211 from ~0.70–0.75 starts) — equal optimization work per
step, so the gap is throughput, not shortcut. verl-omni per-step phases (its own
`timing_s/*`, last step): gen 66.9 + old_log_prob 14.1 + update_actor 28.3 +
reward 1.0 (async-overlapped) ≈ 110 s.

Disclosed differences (inherent to an end-to-end framework comparison):

- **Each framework runs its own pinned rollout stack** — UniRL: vLLM-Omni 0.20.0,
  torch 2.11.0+cu129; verl-omni: vLLM 0.24.0 + vLLM-Omni @ its CI pin `fe478a95` +
  verl @ `8a694930`, torch 2.11.0 (cu13). Neither supports the other's engine version.
- verl's pipeline recomputes `old_log_prob` post-rollout (14.1 s/step); UniRL takes
  SDE log-probs from the engine during rollout. Excluding that stage entirely still
  leaves 97.3 vs 45.0 s/step (2.16×).
- verl-omni overlaps reward asynchronously (counted; 1.0 s wall). SDE noise math is
  its `cps` window vs UniRL's `FlowSDEStrategy` window — same denoise FLOP shape.
  Same prompt files, each side's own seed-42 shuffle order.

Environment notes for reproducing the verl-omni side: hosts with glibc < 2.31 get no
`llguidance` 1.7.x wheel (a vLLM 0.24 dep) — build the sdist with Rust (if `cargo
metadata` hangs on your network, set a crates mirror and `[http] multiplexing = false`);
on driver-535 fleets the cu13 torch stack needs NVIDIA's `cuda-compat-13-*` forward-compat
libs on `LD_LIBRARY_PATH`; `HF_HUB_DISABLE_XET=1` if xet-backed HF downloads stall.

## Hands-on: the aligned Qwen-Image + FlowGRPO OCR pair

One aligned workload, pinned from verl-omni's
[`run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh).
Both sides: 32 prompts/step × 16 samples/prompt @512², 10 denoise steps, true-CFG
4.0, SDE noise on 2 of the first 5 steps (`eta`/`noise_level` 1.2), 2 mini-batch
updates/step (micro 16/GPU), LoRA r64/α128 on the 8 attention projections **plus**
the 4 img/txt MLP projections, lr 3e-4 / wd 1e-4 / clip 1e-5, no KL/ref, FSDP2
CPU offload, the same `datasets/ocr` prompts, 1×4 GPUs, val/save off.

```bash
# 0. one-time: fetch the pinned upstream + build verl-omni's prompt parquet
git submodule update --init benchmarks/speed_benchmarks/verl_omni/upstream
python benchmarks/speed_benchmarks/verl_omni/make_ocr_parquet.py   # → ~/data/ocr/qwen_image

# 1. UniRL side (UniRL env, repo root; needs paddleocr + python-Levenshtein)
QWEN_IMAGE=Qwen/Qwen-Image STEPS=25 GPUS=4 \
  bash benchmarks/speed_benchmarks/verl_omni/run_unirl_qwen_image_aligned.sh 2>&1 | tee unirl_qwen.log
python benchmarks/speed_benchmarks/parse_perf.py unirl_qwen.log --samples-per-step 512 --gpus 4

# 2. verl-omni side (verl-omni env per upstream/docs/start/install.md, same GPUs, one at a time)
QWEN_IMAGE=Qwen/Qwen-Image STEPS=25 ATTN=sdpa GPUS=4 \
  bash benchmarks/speed_benchmarks/verl_omni/run_verlomni_qwen_image_aligned.sh 2>&1 | tee verlomni_qwen.log
python benchmarks/speed_benchmarks/verl_omni/parse_verl_timing.py verlomni_qwen.log --samples-per-step 512 --gpus 4
```

`ATTN=sdpa` is the backend-aligned row (UniRL's diffusers/vLLM-Omni stack runs
SDPA-class kernels); `ATTN=fa3` runs verl-omni's own default attention (FA3 hub
kernel actor + `FLASH_ATTN_3_HUB` rollout) as its best-config row.

Point `VERL_OMNI=` at a local verl-omni checkout if the submodule is not
initialized.

## Measured (Qwen-Image OCR, 1×4 GPUs, 25 steps, first 2 dropped)

| side | attention | median s/step | p90 | samples/GPU-h | rel. |
|---|---|---|---|---|---|
| **UniRL** (`qwen_image_grpo_vllmomni.yaml`, overrides above) | SDPA | — | — | — | — |
| verl-omni | SDPA-aligned | — | — | — | — |
| verl-omni | FA3 (its default) | — | — | — | — |

Paste numbers after a run. verl-omni's published recipe row on 4×H800 (300-step
training, not this 25-step speed cut) is 420 s/step / 0.305 samples/GPU/s with
its FA3 default — use that only as a ballpark, not as a paired number.

Disclosed differences (inherent to an end-to-end framework comparison):

- **Reward backend is not the same scorer.** verl-omni colocates
  `Qwen/Qwen3-VL-8B-Instruct` GenRM OCR (`genrm_ocr.py`, TP=4 on the same 4
  GPUs). UniRL has no colocated Qwen3-VL GenRM; it uses in-process PaddleOCR
  (CPU) with the same quoted-span + Levenshtein formula. Reward GPU time is
  counted on the verl-omni side and essentially free on the UniRL side — do
  not treat samples/GPU-h as a pure train+rollout comparison. A
  time-to-reward-threshold curve with one scorer remains the fairest long-run
  evidence.
- **Each framework runs its own pinned rollout stack** (same class of mismatch
  as the SD3.5 pair). UniRL's Qwen-Image vLLM-Omni stage uses `max_num_seqs=1`;
  verl-omni's recipe packs with `max_num_seqs=8`.
- verl's pipeline recomputes `old_log_prob` post-rollout; UniRL takes SDE
  log-probs from the engine during rollout and anchors `old_logp_source=replay`.
- SDE index selection is verl's contiguous `sde` window of size 2 in `[0,5]`
  vs UniRL's `FlowSDEStrategy` + `AllSDEScheduler` drawing 2 of the first 5
  steps — same denoise FLOP shape.
- Same prompt files (`datasets/ocr`); verl-omni wraps them in a chat-template
  parquet then strips the system turn via `custom_chat_template` so both sides
  encode the raw user line. Each side's own seed-42 shuffle order.

## Other overlapping (model, algorithm) pairs, both natively supported

| pair | UniRL side | verl-omni side |
|---|---|---|
| Qwen-Image + FlowGRPO + OCR | `run_unirl_qwen_image_aligned.sh` | `run_verlomni_qwen_image_aligned.sh` (from `run_qwen_image_ocr_lora.sh`) |
| Wan2.2-T2V-14B + DanceGRPO | `examples/diffusion/wan22/wan22_t2v_14b_dancegrpo.yaml` | its Wan2.2 recipe |

Same protocol when running these: pin the workload from the verl-omni recipe, same
reward behind the same placement, ≥20 steady-state steps, publish both configs and
the disclosure list next to every number. A time-to-reward-threshold curve (same
eval, wall-clock x-axis) remains the strongest end-to-end evidence for long runs.
