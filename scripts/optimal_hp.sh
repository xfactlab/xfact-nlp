for i in $(seq 0 4) # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  bash vessl-create.sh 8 0.00001 pretrain_pt polynomial $i
  bash vessl-create.sh 8 0.0005 pretrain_pthl constant $i
  bash vessl-create.sh 8 0.00005 pretrain_pt cosine_with_restarts $i
  bash vessl-create.sh 8 0.0001 pretrain_hl polynomial $i
done

for i in $(seq 0 4) # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  bash hover-create.sh 8 0.00001 pretrain_pt polynomial $i
  bash hover-create.sh 8 0.0005 pretrain_pthl constant $i
  bash hover-create.sh 8 0.00005 pretrain_pt cosine_with_restarts $i
  bash hover-create.sh 8 0.0001 pretrain_hl polynomial $i
done

for i in $(seq 0 4) # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  bash qg-create.sh 8 0.0005 pretrain_pthl constant $i
done

for i in $(seq 0 4) # 0.000001 0.0000005 0.0000001 0.00000005 0.00000001
do
  bash qg-shuf-create.sh 8 0.0005 pretrain_pthl constant $i
done