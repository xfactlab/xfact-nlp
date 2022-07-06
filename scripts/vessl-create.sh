echo "Which branch do you want to run your program? Press ENTER if you want to use 'HEAD'."
read GIT_REF
if [ "$GIT_REF" == "" ]
  then
    GIT_REF="HEAD"
fi
echo "Using ${GIT_REF}"
NUM_GPUS=${1:-1}
export TRANSFORMERS_CACHE=/cache/transformers
export XDG_CACHE_HOME=/cache/pytorch
vessl experiment create \
  --organization "kaist-jtlab" \
  --project "deardr" \
  --image-url "quay.io/vessl-ai/kernels:ngc.pytorch.22.04-py3" \
  --cluster "cluster-2080" \
  --resource "gpu-${NUM_GPUS}" \
  --command "bash -x scripts/install.sh && bash -x scripts/deardr/pretrain.sh ${NUM_GPUS} \$reader \$learning_rate \$batch_size \$steps \$eval_freq \$lr_scheduler_type" \
  --working-dir /root/deardr --root-volume-size "100Gi" --output-dir "/output/" \
  --git-ref "/root/deardr:github/j6mes-lab/deardr/${GIT_REF}" \
  --dataset "/data/:kaist-jtlab/deardr-dataset" \
  --dataset "/cache/:kaist-jtlab/cache" \
  -h reader=pretrain_hl -h learning_rate=1e-5 -h batch_size=8 -h steps=1 -h eval_freq=400 -h lr_scheduler_type=linear -h DATA_ROOT=/data

