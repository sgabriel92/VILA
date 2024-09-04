#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "master_addr="$(master_addr)
echo "hostnames="$(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"
BASE_MODEL_PATH="Efficient-Large-Model/VILA1.5-3B-s2"
echo "BASE_MODEL_PATH="$BASE_MODEL_PATH

n_nodes=1
OUTPUT="v1_5-3b-s2-ft-test"


torchrun --nnodes=$n_node --nproc_per_node=1 --master_port=25001 \
    --master_addr "127.0.0.1" --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version v1 \
    --data_path ../LLaVA/armbench/train/dataset.json \
    --validation_data_path ../LLaVA/armbench/validation/dataset.json \
    --image_folder ../LLaVA/armbench/images/ \
    --vision_tower google/siglip-so400m-patch14-384 \
    --s2 True \
    --s2_scales "384,768" \
    --s2_max_split_size 384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 0.1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
