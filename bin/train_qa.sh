export PYTHONPATH=$PWD
for fold_id in 1 2 3 4 5 6 7
do
        python scripts/main.py \
                --experiment "qa" \
                --output_dir "outputs/qa/fold_$fold_id" \
                --model_name_or_path "roberta-base" \
                --tokenizer "roberta-base" \
                --loss_fct "online-constrastive" \
                --distance "cosine" \
                --directory "kpm_k_folds/fold_$fold_id" \
                --test_directory "kpm_k_folds/test/" \
                --logging_dir "runs/qa/fold_$fold_id" \
                --logging_steps 40 \
                --overwrite_output_dir \
                --num_train_epochs 15 \
                --early_stop 5 \
                --train_batch_size 128 \
                --val_batch_size 128 \
                --do_train \
                --evaluate_during_training \
                --warmup_steps 0 \
                --gradient_accumulation_steps 1 \
                --learning_rate 0.00003 \
                --margin 0.3 \
                --drop_rate 0.1 \
                --n_hiddens 4 \
                --max_topic_length 10 \
                --max_statement_length 45 \
                --stance_dim 32 \
                --text_dim 256 \
                --num_workers 4 \
                --seed 0 
done