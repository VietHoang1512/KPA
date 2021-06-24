import os

import torch
from transformers import AutoTokenizer

from src.question_answering.data_argument import QADataArguments
from src.question_answering.datasets import QADataset
from src.question_answering.model_argument import QAModelArguments
from src.question_answering.models import QABertModel
from src.train_utils.helpers import count_parameters, seed_everything
from src.train_utils.trainer import Trainer
from src.train_utils.training_argument import TrainingArguments
from src.utils.data import get_data, prepare_inference_data
from src.utils.hf_argparser import HfArgumentParser
from src.utils.logging import custom_logger
from src.utils.signature import print_signature

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = custom_logger(__name__)

if __name__ == "__main__":

    print_signature()

    parser = HfArgumentParser((QAModelArguments, QADataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if torch.cuda.device_count() >= 1:
        logger.info(f"Device {torch.cuda.get_device_name(0)} is availble")

    logger.warning("Device: %s", training_args.device)
    seed_everything(training_args.seed)

    if model_args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer, use_fast=False)
    elif model_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer"
        )

    model = QABertModel(args=model_args)
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    logger.info(f"Number of parameters: {count_parameters(model)}")

    train_df, train_arg_df, train_kp_df, train_labels_df = get_data(gold_data_dir=data_args.directory, subset="train")
    val_df, val_arg_df, val_kp_df, val_labels_df = get_data(gold_data_dir=data_args.directory, subset="dev")
    test_df, test_arg_df, test_kp_df, test_labels_df = get_data(gold_data_dir=data_args.test_directory, subset="test")

    val_inf_df = prepare_inference_data(val_arg_df, val_kp_df)
    test_inf_df = prepare_inference_data(test_arg_df, test_kp_df)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    train_dataset = QADataset(
        df=train_df,
        arg_df=train_arg_df,
        labels_df=train_labels_df,
        tokenizer=tokenizer,
        args=data_args,
    )
    val_dataset = QADataset(
        df=val_inf_df,
        arg_df=val_arg_df,
        labels_df=val_labels_df,
        tokenizer=tokenizer,
        args=data_args,
    )

    test_dataset = QADataset(
        df=test_inf_df,
        arg_df=test_arg_df,
        labels_df=test_labels_df,
        tokenizer=tokenizer,
        args=data_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
    if training_args.do_inference:
        trainer.inference(test_dataset)
