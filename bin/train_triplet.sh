export PYTHONPATH=$PWD
for fold_id in 1 2 3 4 
do
        echo "TRAINING ON FOLD $fold_id"
        python scripts/main.py \
                --experiment "triplet" \
                --output_dir "outputs/triplet/fold_$fold_id" \
                --model_name_or_path "roberta-base" \
                --tokenizer "roberta-base" \
                --distance "cosine" \
                --sample_selection "neg" \
                --directory "kpm_k_folds/fold_$fold_id" \
                --test_directory "kpm_k_folds/test/" \
                --logging_dir "runs/triplet/fold_$fold_id" \
                --logging_steps 20 \
                --overwrite_output_dir \
                --num_train_epochs 100\
                --early_stop 20 \
                --train_batch_size 96 \
                --val_batch_size 96 \
                --do_train \
                --evaluate_during_training \
                --warmup_steps 0 \
                --gradient_accumulation_steps 1 \
                --learning_rate 0.00003 \
                --margin 0.3 \
                --drop_rate 0.1 \
                --n_hiddens 3 \
                --max_len 30 \
                --argument_max_len 50 \
                --stance_dim 32 \
                --text_dim 256 \
                --num_workers 4 \
                --seed 0
done
