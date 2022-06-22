MASTER_PORT=50123 tsp bash scripts/deardr/pretrain_fine.sh 1 pretrain_pthl_filter constant_with_warmup 5e-05 0.0 0.2 16 4 150
MASTER_PORT=50124 tsp bash scripts/deardr/pretrain_fine.sh 1 pretrain_pt constant_with_warmup 5e-06 0.0 0.3 16 4 150
MASTER_PORT=50125 tsp bash scripts/deardr/pretrain_fine.sh 1 pretrain_pthl constant_with_warmup 5e-06 0.0 0.1 16 4 150
