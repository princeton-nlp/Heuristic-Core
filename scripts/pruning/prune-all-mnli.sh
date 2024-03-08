#!/bin/bash

# For the effective size experiments, you'll have to prune over a grid of sparsities
# We used 
# for seed in 42 43 44; do
# for sparsity in 5 10 15 20 25 30 35 40 45 50 55 60 65 70; do 
# for steps in 500 6500 12500 18500 24500 30500 36500 42500 48500 54500; do
# for MNLI, and
# for seed in 42 43 44; do
# for sparsity in 5 10 15 20 25 30 35 40 45 50 55 60 65; do 
# for steps in 500 2000 3500 5000 6500 8000 9500 11000 12500 14000; do
# These require very little modification to the script below.

for seed in 42 43 44 45 46 47 48 49 50 51 52 53; do

sparsity=50                                # 50 = 0.5, etc. Note that sparsity = percent of heads/MLPS *missing*, not kept
warmup=$(( 6500 + 50 * $sparsity ))        # This is how many steps we will ramp up the sparsity over
complete=$(( $warmup + 6 * $sparsity ))    # Total number of steps to train for

# Wrap an sbatch script here, if you want
# Activate any environments you need here

# Note: we used evaluation/checkpointing every 64 steps, but you can do 512 as below, 
# to save time.

WANDB_MODE=disabled python src/pruning/prune-glue.py \
    --do_train \
    --do_eval \
    --glue_path glue \
    --task_name mnli \
    --initialize_from ./models/ft/MNLI-61360/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.1 \
    --reg_learning_rate 1 \
    --max_steps $complete \
    --warmup_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 512 \
    --save_steps 512 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_sparsity 0.0 \
    --target_sparsity 0.${sparsity} \
    --num_sparsity_warmup_steps $warmup \
    --max_train_samples 1000000 \
    --max_eval_samples 2000 \
    --output_dir ./models/pruned/mnli-seed${seed}mean-${sparsity}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 32 \
    --eval_accumulation_steps 16 \
    --warmup_type linear \
    --avg_activation_path ./data/activations/mnli.pkl \
    --seed ${seed} \
    --dataloader_pin_memory false

done