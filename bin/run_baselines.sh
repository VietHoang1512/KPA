#!bin/bash
export PYTHONPATH=/content/KPA
python src/scripts/run_baselines.py \
        --output_dir "outputs" \
        --model_name "roberta-base" \
        --directory "kpm_data" \
        --overwrite_output_dir \
        --num_train_epochs 15 \
        --early_stop 5 \
        --train_batch_size 64 \
        --val_batch_size 64 \
        --do_train \
        --evaluate_during_training \
        --warmup_steps 250 \
        --gradient_accumulation_steps 2 \
        --learning_rate 0.00003 \
        --margin 1.0 \
        --drop_rate 0.15 \
        --n_hiddens 2 \
        --max_len 36 \
        --argument_max_len 64 \
        --stance_dim 32 \
        --text_dim 256 \
        --num_workers 2 \
        --seed 1512