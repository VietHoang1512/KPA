export PYTHONPATH=$PWD
for fold_id in 1 2 3 4 5 6
do
        python src/scripts/train_qa.py \
                --output_dir "outputs/qa/fold_$fold_id" \
                --model_name_or_path "roberta-base" \
                --tokenizer "roberta-base" \
                --loss_fct "online-constrastive" \
                --distance "cosine" \
                --directory "kpm_6_folds/fold_$fold_id" \
                --test_directory "kpm_6_folds/test/" \
                --logging_dir "runs/qa/fold_$fold_id" \
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
                --margin 0.5 \
                --drop_rate 0.1 \
                --n_hiddens -1 \
                --max_topic_length 16 \
                --max_statement_length 48 \
                --stance_dim 32 \
                --text_dim 256 \
                --num_workers 8 \
                --seed 0
done
