#!/usr/bin/env bash
# verl-omni side of the aligned Qwen-Image+FlowGRPO OCR speed pair (see README.md).
# Pins examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh with
# val/save off, a fixed step budget, and the same ATTN toggle as the SD3.5 pair.
# Run inside a verl-omni environment (install per upstream/docs/start/install.md
# against the pinned submodule), 1x4 GPUs. Data first:
#   python ../make_ocr_parquet.py
# Then:
#   QWEN_IMAGE=<hf-id-or-local-dir> STEPS=25 ATTN=sdpa GPUS=4 \
#     bash run_verlomni_qwen_image_aligned.sh
# ATTN=sdpa is the backend-aligned row (matches UniRL's SDPA-class kernels);
# ATTN=fa3 is verl-omni's own default/best attention config (FA3 hub).
# Parse: python ../parse_verl_timing.py <log> --samples-per-step 512 --gpus 4
set -ex
cd "${VERL_OMNI:-$(dirname "$0")/upstream}"

QWEN_IMAGE=${QWEN_IMAGE:-Qwen/Qwen-Image}
REWARD_MODEL=${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DATA=${DATA:-$HOME/data/ocr/qwen_image}
STEPS=${STEPS:-25}
GPUS=${GPUS:-4}
REWARD_TP=${REWARD_TP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}

if [ $((GPUS % REWARD_TP)) -ne 0 ] || [ $((GPUS % ROLLOUT_TP)) -ne 0 ]; then
  echo "GPUS=$GPUS must be divisible by REWARD_TP=$REWARD_TP and ROLLOUT_TP=$ROLLOUT_TP" >&2
  exit 1
fi

if [ "${ATTN:-sdpa}" = "fa3" ]; then
  ACTOR_ATTN=_flash_3_varlen_hub; ROLLOUT_ATTN=FLASH_ATTN_3_HUB
else
  ACTOR_ATTN=native; ROLLOUT_ATTN=TORCH_SDPA
fi
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/test.parquet \
    data.train_batch_size=32 \
    data.val_max_samples=8 \
    data.max_prompt_length=256 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.path=$QWEN_IMAGE \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    actor_rollout_ref.model.attn_backend=$ACTOR_ATTN \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=${REQUEST_BATCH_MAX_WAIT_MS} \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=$((GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$REWARD_MODEL \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console"]' \
    trainer.project_name=speed_benchmarks \
    trainer.experiment_name=qwen_image_flowgrpo_ocr_aligned \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS "$@"
