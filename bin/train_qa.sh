export PYTHONPATH=$PWD
python src/scripts/train_qa.py \
        --output_dir "outputs" \
        --model_name_or_path "roberta-base" \
        --tokenizer "roberta-base" \
        --loss_fct "online-constrastive" \
        --distance "euclidean" \
        --directory "kpm_data" \
        --logging_dir "runs" \
        --overwrite_output_dir \
        --num_train_epochs 15 \
        --early_stop 5 \
        --train_batch_size 32 \
        --val_batch_size 32 \
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