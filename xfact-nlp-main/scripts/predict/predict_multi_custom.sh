TASK=$1
MODEL_PATH=$2
INPUT_FILE=$3
OUTPUT_FILE=$4


python -m deardr.predict \
    --model_name_or_path $MODEL_PATH \
    --dataset_reader deardr \
    --validation_frontend_reader $TASK \
    --test_file $INPUT_FILE \
    --out_file $OUTPUT_FILE