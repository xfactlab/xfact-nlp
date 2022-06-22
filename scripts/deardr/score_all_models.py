import os.path
from glob import glob

if __name__ == "__main__":
    files = glob("../deardr_work/**/*dev*.jsonl",recursive=True)

    for file in files:

        print(f"python -m deardr.score {file}")