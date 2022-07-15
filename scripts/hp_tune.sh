#!/bin/bash
gpu=2
for i in 0.00005 0.00001 0.000005 # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  for j in "pretrain_hl" "pretrain_pthl" "pretrain_pt"
  do
    for k in "cosine_with_restarts" "polynomial" "constant" "constant_with_warmup" # "constant" "linear" #
    do
      bash vessl-create.sh $gpu $i $j $k
    done
  done
done
