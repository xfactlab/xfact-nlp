#!/bin/bash
start=0.00005
end=0.000005
factor=-0.000005
gpu=4
for i in 0.00005 0.00001 0.000005 0.000001 # 0.0000005 0.0000001 0.00000005 0.00000001
do
  for j in "pretrain_hl" "pretrain_pthl" "pretrain_pt"
  do
    for k in "constant" "linear" #"cosine_with_restarts" "polynomial" "constant" "constant_with_warmup"
    do
      bash vessl-create.sh $gpu $i $j $k
    done
  done
done
