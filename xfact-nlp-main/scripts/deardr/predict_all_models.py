import os.path
from glob import glob

if __name__ == "__main__":
    models = glob("../deardr_work/finetune_ablation/**/pytorch_model.bin",recursive=True)

    print(models)
    for model in models:
        if "checkpoint" in model:
            continue

        basename = os.path.dirname(model)
        print(f"tsp bash scripts/predict.sh fever_no_nei {basename} data/fever/dev.jsonl")