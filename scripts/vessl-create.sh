NUM_GPUS=${1:-1}

vessl experiment create \
  --organization "kaist-jtlab" \
  --project "deardr" \
  --image-url "j6mes/dev-env:build-21" \
  --cluster "cluster-2080" \
  --resource "gpu-${NUM_GPUS}" \
  --command "bash -x scripts/deardr/pretrain.sh ${NUM_GPUS} \$reader \$learning_rate \$batch_size \$steps \$eval_freq \$lr_scheduler_type" \
  --working-dir /root/deardr --root-volume-size "100Gi" --output-dir "/output/" \
  --dataset "/deardr/:kaist-jtlab/deardr" \
  --git-ref "/root/deardr:github/j6mes/deardr/HEAD" \
  --dataset "/fever/:kaist-jtlab/fever" \
  -h reader=pretrain_hl -h learning_rate=1e-5 -h batch_size=8 -h steps=1 -h eval_freq=400 -h lr_scheduler_type=linear

