#!bin/bash
export PYTHONPATH=$PWD
python src/scripts/run_baselines.py \
        --output_dir "outputs" \
        --model_name "roberta-base" \
        --directory "kpm_data" \
        --logging_dir "runs" \
        --overwrite_output_dir \
        --num_train_epochs 15 \
        --early_stop 5 \
        --train_batch_size 64 \
        --val_batch_size 64 \
        --do_train \
        --evaluate_during_training \
        --warmup_steps 0 \
        --gradient_accumulation_steps 1 \
        --learning_rate 0.00003 \
        --margin 1.0 \
        --drop_rate 0.0 \
        --n_hiddens 4 \
        --max_len 24\
        --argument_max_len 48 \
        --stance_dim 32 \
        --text_dim 256 \
        --num_workers 2 \
        --seed 1512