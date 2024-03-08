## You might want to wrap this in a slurm script or something similar

TASK_NAME=MNLI      # QQP for QQP
SEED=42     
STEPS=61360         # 40000 for QQP
BATCH_SIZE=32       # 128 for QQP via gradient accumulation

WANDB_MODE=disabled python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --max_steps $STEPS \
  --output_dir models/ft/${TASK_NAME}-${STEPS}/ \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 500 \
  --seed $SEED