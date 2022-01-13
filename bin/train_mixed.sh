export PYTHONPATH=$PWD
for fold_id in 1 2 3 4 5 6 7
do
        echo "TRAINING ON FOLD $fold_id"
        python scripts/main.py \
                --experiment "mixed" \
                --output_dir "outputs/mixed/fold_$fold_id" \
                --model_name_or_path "roberta-base" \
                --tokenizer "roberta-base" \
                --loss_fct "online-constrastive" \
                --distance "cosine" \
                --sample_selection "neg" \
                --directory "kpm_k_folds/fold_$fold_id" \
                --test_directory "kpm_k_folds/test/" \
                --logging_dir "runs/mixed/fold_$fold_id" \
                --logging_steps 10 \
                --overwrite_output_dir \
                --num_train_epochs 50\
                --early_stop 20 \
                --train_batch_size 128 \
                --val_batch_size 128 \
                --do_train \
                --evaluate_during_training \
                --warmup_steps 0 \
                --gradient_accumulation_steps 1 \
                --learning_rate 0.00003 \
                --pair_margin 0.3 \
                --triplet_margin 0.3 \
                --drop_rate 0.2 \
                --n_hiddens 4 \
                --max_len 30 \
                --argument_max_len 50 \
                --stance_dim 32 \
                --text_dim 256 \
                --num_workers 4 \
                --seed 0 \
                --normalize 
done