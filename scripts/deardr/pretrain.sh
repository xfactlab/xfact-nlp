NUM_GPUS=$1
train_reader=$2
learning_rate=$3
batch_size=$4
steps=$5
eval_freq=$6
scheduler=$7
val_reader=$8
train_db=$9
val_db=${10}
train_file=${11}
val_file=${12}
model_name=${13}
log_freq=50
data_root=${DATA_ROOT:-/}

export PYTHONPATH=src

#python src/deardr/train.py \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
  src/deardr/train.py \
  --project_name DEARDR-ZEROSHOT \
  --dataset_reader deardr \
  --validation_dataset_reader deardr \
  --train_frontend_reader $train_reader \
  --validation_frontend_reader $val_reader \
  --model_name_or_path $model_name \
  --output_dir /output \
  --train_file ${data_root}/${train_db}/${train_file} \
  --validation_file ${data_root}/${val_db}/$val_file \
  --prefix_path ${data_root}/prefix-tree/wikipedia-titles-structured-pt.pkl \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --logging_steps $log_freq \
  --save_steps $eval_freq \
  --eval_steps $eval_freq \
  --num_train_epochs 8 \
  --save_total_limit 3 \
  --max_eval_samples 1000 \
  --load_best_model_at_end \
  --metric_for_best_model macro_r_precision \
  --evaluation_strategy steps \
  --learning_rate $learning_rate \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps $steps \
  --lr_scheduler_type $scheduler

