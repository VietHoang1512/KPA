#!bin/bash
export PYTHONPATH=$PWD
python src/bert/hf_argparser.py --model_name "roberta-base" --config_name "roberta-base" --tokenizer "roberta-base" --directory "kpm_data" --output_dir "outputs"
 