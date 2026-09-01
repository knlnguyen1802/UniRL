#!/usr/bin/env bash
# UniRL side of the aligned Qwen-Image+FlowGRPO OCR speed pair (see README.md).
# Pins the workload from verl-omni's
#   examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh
# Run from the UniRL repo root, inside a UniRL environment, 1x4 GPUs.
#   QWEN_IMAGE=<hf-id-or-local-dir> STEPS=25 GPUS=4 \
#     bash benchmarks/speed_benchmarks/verl_omni/run_unirl_qwen_image_aligned.sh
# Then: python benchmarks/speed_benchmarks/parse_perf.py <log> --samples-per-step 512 --gpus 4
set -ex
export QWEN_IMAGE_PATH=${QWEN_IMAGE:-Qwen/Qwen-Image}
export REPORT_TO_WANDB=${REPORT_TO_WANDB:-false}
GPUS=${GPUS:-4}

python -m unirl.train_diffusion --config-name=diffusion/qwen_image/qwen_image_grpo_vllmomni \
  num_devices=${GPUS} \
  ++devices_per_node=${GPUS} \
  batch_size=32 sampling.samples_per_prompt=16 \
  sampling.height=512 sampling.width=512 sampling.num_inference_steps=10 \
  sampling.guidance_scale=4.0 sampling.eta=1.2 \
  sampling.scheduler.num_sde_steps=2 \
  'sampling.scheduler.timestep_fraction=[0,0.5]' \
  bundle.config.max_sequence_length=256 \
  pipeline.max_sequence_length=256 \
  backend.optimizer_cfg.learning_rate=3e-4 backend.optimizer_cfg.weight_decay=1e-4 \
  backend.fsdp_cfg.cpu_offload=true \
  algorithm.clip_range=1e-5 \
  stack.micro_batch_size=16 \
  backend.lora_cfg.rank=64 backend.lora_cfg.alpha=128 \
  'backend.lora_cfg.target_modules=[to_q,to_k,to_v,"to_out.0",add_q_proj,add_k_proj,add_v_proj,to_add_out,"img_mlp.net.0.proj","img_mlp.net.2","txt_mlp.net.0.proj","txt_mlp.net.2"]' \
  '~reward.backend' \
  +reward.backend._target_=unirl.reward.local.ocr.OCRRewardScorer \
  +reward.backend.base_device=cuda \
  +reward.backend.config._target_=unirl.reward.local.ocr.OCRSpec \
  data_source.args.run.data_path=datasets/ocr/train.txt \
  data_source.args.run.eval_data_path=datasets/ocr/test.txt \
  num_rollouts=${STEPS:-25} "$@"
