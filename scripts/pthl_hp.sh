for i in $(seq 0 4) # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  bash vessl-create.sh 8 0.0005 pretrain_pthl constant $i
done
