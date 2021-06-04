import logging
import os

from transformers import AutoTokenizer

from src.bert.data_argument import DataArguments
from src.bert.datasets import BertKPADataset
from src.bert.model_argument import ModelArguments
from src.bert.models import BertKPAModel
from src.bert.trainer import Trainer
from src.bert.training_argument import TrainingArguments
from src.utils.data import get_data, prepare_inference_data
from src.utils.hf_argparser import HfArgumentParser
from src.utils.train_utils import seed_everything

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("Device: %s", training_args.device)
    seed_everything(training_args.seed)

    if model_args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer)
    elif model_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    model = BertKPAModel(
        bert_model=model_args.model_name,
        n_hiddens=model_args.n_hiddens,
        stance_dim=model_args.stance_dim,
        text_dim=model_args.text_dim,
    )

    train_df, train_arg_df, train_kp_df, train_labels_df = get_data(gold_data_dir="kpm_data", subset="train")
    val_df, val_arg_df, val_kp_df, val_labels_df = get_data(gold_data_dir="kpm_data", subset="dev")

    train_inf_df = prepare_inference_data(train_arg_df, train_kp_df)
    val_inf_df = prepare_inference_data(val_arg_df, val_kp_df)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    train_dataset = BertKPADataset(
        df=train_df,
        arg_df=train_arg_df,
        labels_df=train_labels_df,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        argument_max_len=data_args.argument_max_len,
    )
    val_dataset = BertKPADataset(
        df=val_df,
        arg_df=val_arg_df,
        labels_df=val_labels_df,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        argument_max_len=data_args.argument_max_len,
    )
    train_inf_dataset = BertKPADataset(
        df=train_inf_df,
        arg_df=train_arg_df,
        labels_df=train_labels_df,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        argument_max_len=data_args.argument_max_len,
    )
    val_inf_dataset = BertKPADataset(
        df=val_inf_df,
        arg_df=val_arg_df,
        labels_df=val_labels_df,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        argument_max_len=data_args.argument_max_len,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_inf_dataset=train_inf_dataset,
        val_inf_dataset=val_inf_dataset,
    )
    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
