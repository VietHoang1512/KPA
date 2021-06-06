#!bin/bash
export PYTHONPATH=/content/KPA
python src/scripts/run_baselines.py \
        --num_train_epochs 15 \
        --margin 1.0 \
        --learning_rate 0.00003 \
        --output_dir "outputs" \
        --model_name "roberta-base" \
        --n_hiddens 4 \
        --do_train \
        --drop_rate 0.0 \
        --evaluate_during_training \
        --overwrite_output_dir \
        --max_len 64 \
        --argument_max_len 64 \
        --train_batch_size 64 \
        --val_batch_size 64 \
        --stance_dim 16 \
        --text_dim 256