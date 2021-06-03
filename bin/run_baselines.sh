#!bin/bash
export PYTHONPATH=$PWD
python src/scripts/run_baselines.py --output_dir "outputs" --model_name "roberta-base" --do_train --evaluate_during_training --overwrite_output_dir --max_len 24  --argument_max_len 48 --train_batch_size 32 --val_batch_size 48
