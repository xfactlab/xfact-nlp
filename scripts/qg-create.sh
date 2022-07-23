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
SEED=${5}

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
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=75 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=kilt -h train_db=wiki-pretraining -h val_db=kilt -h train_file=qg_filtered_10k.jsonl -h val_file=triviaqa-dev-10k-kilt.jsonl \
  -h model_name=t5-base -h seed=${SEED} -h epoch=5 -h train_beam=10 -h eval_beam=10 \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch


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
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=75 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=kilt -h train_db=wiki-pretraining -h val_db=kilt -h train_file=qg_filtered_10k.jsonl -h val_file=triviaqa-dev-k-kilt.jsonl \
  -h model_name=t5-base -h seed=${SEED} -h epoch=5 -h train_beam=10 -h eval_beam=10 \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch

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
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=75 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=kilt -h train_db=wiki-pretraining -h val_db=kilt -h train_file=qg_filtered_10k.jsonl -h val_file=nq-dev-kilt-1000.jsonl \
  -h model_name=t5-base -h seed=${SEED} -h epoch=5 -h train_beam=10 -h eval_beam=10 \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch


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
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=75 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=kilt -h train_db=wiki-pretraining -h val_db=kilt -h train_file=qg_filtered_10k.jsonl -h val_file=hotpotqa-dev-kilt-10000.jsonl \
  -h model_name=t5-base -h seed=${SEED} -h epoch=5 -h train_beam=10 -h eval_beam=10 \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch


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
  -h reader=$READER_TRAIN -h learning_rate=$LEARNING_RATE -h batch_size=8 -h steps=1 -h eval_freq=75 -h lr_scheduler_type=${LR_SCHEDULER} \
  -h val_reader=kilt -h train_db=wiki-pretraining -h val_db=kilt -h train_file=qg_filtered_10k.jsonl -h val_file=hotpotqa-dev-kilt-1000.jsonl \
  -h model_name=t5-base -h seed=${SEED} -h epoch=5 -h train_beam=10 -h eval_beam=10 \
  -h DATA_ROOT=/data -h TRANSFORMERS_CACHE=/cache/transformers -h XDG_CACHE_HOME=/cache/pytorch
