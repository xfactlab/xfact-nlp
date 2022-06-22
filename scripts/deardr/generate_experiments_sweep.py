import math
import random

if __name__ == "__main__":

    NUM_GPUS = 4
    scheduler = "constant_with_warmup"

    experiments = []
    for reader in ["pretrain_pt","pretrain_hl","pretrain_pthl"]:
        for bs in [16]:
            for steps in [1]:
                for lr in [5e-6]:
                    for decay in [0.0]:
                        for dropout in [0.1]:
                            experiments.append((reader, lr, decay, dropout, bs, steps))

    print(len(experiments))
    random.shuffle(experiments)



    for reader, lr, decay, dropout, batch_size, steps in experiments:


        """
            Format:
                NUM_GPUS 
                reader
                scheduler
                learning_rate
                weight_decay
                dropout
                batch_size
                steps
                eval_freq
        """
        size = int(4/NUM_GPUS)
        print(f"scripts/deardr/pretrain.sh {NUM_GPUS} {reader} {scheduler} {lr} {decay} {dropout} 16 {size} 2500")
