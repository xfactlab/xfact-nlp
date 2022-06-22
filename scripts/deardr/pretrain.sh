NUM_GPUS=$1
reader=$2
scheduler=$3
learning_rate=$4
weight_decay=$5
dropout=$6
batch_size=$7
steps=$8
eval_freq=$9
log_freq=100

export PYTHONPATH=src

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
  src/deardr/train.py \
  --project_name DEARDR-ZEROSHOT \
  --dataset_reader deardr \
  --validation_dataset_reader deardr \
  --train_frontend_reader $reader \
  --validation_frontend_reader fever_no_nei \
  --model_name_or_path t5-base \
  --output_dir ../deardr_work/experiments/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/reader=$reader,task=fever,lr=$learning_rate,weight_decay=$weight_decay,dropout=$dropout,batch_size=$batch_size,steps=$steps \
  --train_file data/pretrain/train.jsonl \
  --validation_file data/fever/dev.jsonl \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --logging_steps $log_freq \
  --save_steps $eval_freq \
  --eval_steps $eval_freq \
  --num_train_epochs 3 \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model macro_r_precision \
  --evaluation_strategy steps \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps $steps \
  --lr_scheduler_type $scheduler

