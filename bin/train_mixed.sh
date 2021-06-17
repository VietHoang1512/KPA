export PYTHONPATH=$PWD
for fold_id in 1 2 3 4 5 6
do
        echo "TRAINING ON FOLD $fold_id"
        python src/scripts/train_mixed.py \
                --output_dir "outputs/mixed/fold_$fold_id" \
                --model_name_or_path "roberta-base" \
                --tokenizer "roberta-base" \
                --loss_fct "online-constrastive" \
                --sample_selection "both" \
                --distance "cosine" \
                --directory "kpm_6_folds/fold_$fold_id" \
                --logging_dir "runs/mixed/fold_$fold_id" \
                --overwrite_output_dir \
                --num_train_epochs 15 \
                --early_stop 5 \
                --train_batch_size 8 \
                --val_batch_size 8 \
                --do_train \
                --evaluate_during_training \
                --warmup_steps 0 \
                --gradient_accumulation_steps 1 \
                --learning_rate 0.00003 \
                --pair_margin 0.5 \
                --triplet_margin 0.5 \
                --drop_rate 0.1 \
                --n_hiddens -1 \
                --max_len 30 \
                --argument_max_len 50 \
                --stance_dim 32 \
                --text_dim 256 \
                --num_workers 2 \
                --seed 0
done