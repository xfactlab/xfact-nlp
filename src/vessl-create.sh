#echo "Which reference_id (ex. 6eb0bc12ac08ff95a68e61785001ac7270dfd62d) do you want to use? Press ENTER if you want to use 'HEAD'."
#read GIT_REF
GIT_REF=""
if [ "$GIT_REF" == "" ]
  then
    GIT_REF="HEAD"
fi
echo "Using ${GIT_REF}"
NUM_GPUS=${1:-1}
LEARNING_RATE=${2}
READER_TRAIN=${3}
LR_SCHEDULER=${4}
vessl experiment create \
  --organization "kaist-jtlab" \
  --project "deardr" \
  --image-url "quay.io/vessl-ai/kernels:ngc.pytorch.22.04-py3" \
  --cluster "cluster-2080" \
  --resource "gpu-${NUM_GPUS}" \
  --command "bash -x scripts/install.sh && bash -x scripts/deardr/pretrain.sh ${NUM_GPUS} \$reader \$learning_rate \$batch_size \$steps \$eval_freq \$lr_scheduler_type \$val_reader \$train_db \$val_db \$train_file \$val_file \$model_name" \
  --working-dir /root/deardr --root-volume-size "100Gi" --output-dir "/output/" \
  --git-ref "/root/deardr:github/j6mes-lab/deardr/${GIT_REF}" \
  --dataset "/data/:kaist-jtlab/deardr-dataset" \
  --dataset "/cache/:kaist-jtlab/cache" \
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=400 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=hover -h train_db=wiki-pretraining -h val_db=hover -h train_file=shuf_10k.jsonl -h val_file=hover-dev.json \
  -h model_name=t5-base \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch