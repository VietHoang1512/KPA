#!bin/bash
# linux
export PYTHONPATH=$PWD
python python src/scripts/run_baselines.py --output_dir "tmp" --model_name "roberta-base" --do_train --overwrite_output_dir --max_len 6 --argument_max_len 12 --train_batch_size 16 --val_batch_size 4


 