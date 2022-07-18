reader=pretrain_pthl
learning_rate=0.0005
batch_size=8
steps=1
eval_freq=400
scheduler=linear
val_reader=fever
train_db=wiki-pretraining
val_db=fever
train_file=shuf_10k.jsonl
val_file=shared_task_dev.jsonl
model_name=checkpoint
log_freq=50
data_root=data
export PYTHONPATH=src
#python src/deardr/train.py \
python src/deardr/predict.py \
  --dataset_reader deardr \
  --model_name_or_path $model_name \
  --test_file ${data_root}/${val_db}/${val_file} \
  --validation_frontend_reader $val_reader \
  --out_file out.jsonl
