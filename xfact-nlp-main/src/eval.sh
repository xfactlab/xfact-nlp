reader=pretrain_pthl
learning_rate=0.0005
batch_size=1
steps=8
eval_freq=400
scheduler=constant
val_reader=kilt
train_db=wiki-pretraining
val_db=kilt
train_file=qg_shuf_10k.jsonl
val_file=triviaqa-dev-k-kilt.jsonl
model_name=t5-base
log_freq=50
data_root=/home/james/data
export PYTHONPATH=src
#python src/deardr/train.py \
python src/deardr/predict.py \
  --dataset_reader deardr \
  --model_name_or_path $model_name \
  --test_file ${data_root}/${val_db}/${val_file} \
  --validation_frontend_reader $val_reader \
  --out_file out.jsonl
