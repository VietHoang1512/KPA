export PYTHONPATH=$PWD

for model in "roberta-base" "bert-base-uncased" 
do
        for seed in 0 1 2 3 4 5
        do
                OUTPUT_DIR=outputs/pseudo_label/$model/$seed/

                mkdir -p $OUTPUT_DIR

                cp src/pseudo_label/models.py  $OUTPUT_DIR

                for fold_id in 1 2 3 4 5 6
                do
                        echo "TRAINING ON FOLD $fold_id"
                        python src/scripts/train_pseudo_label.py \
                                --output_dir "$OUTPUT_DIR/fold_$fold_id" \
                                --model_name_or_path $model \
                                --tokenizer $model \
                                --distance "cosine" \
                                --directory "kpm_6_folds/fold_$fold_id" \
                                --test_directory "kpm_6_folds/test/" \
                                --logging_dir "$OUTPUT_DIR/fold_$fold_id" \
                                --logging_steps 20 \
                                --max_pos 30 \
                                --max_neg 90\
                                --max_unknown 15 \
                                --overwrite_output_dir \
                                --num_train_epochs 5 \
                                --early_stop 10 \
                                --train_batch_size 1 \
                                --val_batch_size 128 \
                                --do_train \
                                --evaluate_during_training \
                                --warmup_steps 0 \
                                --gradient_accumulation_steps 1 \
                                --learning_rate 0.00003 \
                                --margin 0.3 \
                                --drop_rate 0.2 \
                                --n_hiddens 4 \
                                --max_len 30 \
                                --statement_max_len 50 \
                                --stance_dim 32 \
                                --text_dim 256 \
                                --num_workers 4 \
                                --seed $seed \
                                --normalize
                        
                done

                echo "INFERENCE"
                python src/scripts/train_pseudo_label.py \
                        --output_dir "$OUTPUT_DIR" \
                        --model_name_or_path $model \
                        --tokenizer $model \
                        --distance "cosine" \
                        --directory "kpm_6_folds/fold_0/" \
                        --test_directory "kpm_6_folds/test/" \
                        --logging_dir "$OUTPUT_DIR/" \
                        --logging_steps 20 \
                        --max_pos 30 \
                        --max_neg 90\
                        --max_unknown 15 \
                        --overwrite_output_dir \
                        --num_train_epochs 5 \
                        --early_stop 10 \
                        --train_batch_size 1 \
                        --val_batch_size 128 \
                        --do_inference \
                        --evaluate_during_training \
                        --warmup_steps 0 \
                        --gradient_accumulation_steps 1 \
                        --learning_rate 0.00003 \
                        --margin 0.3 \
                        --drop_rate 0.2 \
                        --n_hiddens 4 \
                        --max_len 30 \
                        --statement_max_len 50 \
                        --stance_dim 32 \
                        --text_dim 256 \
                        --num_workers 4 \
                        --seed $seed \
                        --normalize

        done
done