import os
from typing import List

import torch
import yaml
from transformers import AutoTokenizer

from src.backbone.base_arguments import BaseDataArguments, BaseModelArguments
from src.train_utils.helpers import count_parameters, seed_everything
from src.train_utils.trainer import Trainer
from src.train_utils.training_argument import TrainingArguments
from src.utils.data import get_data, length_plot, prepare_inference_data
from src.utils.hf_argparser import HfArgumentParser
from src.utils.logging import custom_logger
from src.utils.signature import print_signature

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = custom_logger(__name__)


def word_len(texts: List[str]) -> List:
    def _word_len(text: str, separator=" ") -> int:
        return len(text.split(separator))

    return list(map(_word_len, texts))


def char_len(texts: List[str]) -> List:
    def _char_len(text: str) -> int:
        return len(text)

    return list(map(_char_len, texts))


def token_len(texts: List[str], tokenizer: AutoTokenizer) -> List:
    def _token_len(text: str) -> int:
        return len(tokenizer.encode(text))

    return list(map(_token_len, texts))


if __name__ == "__main__":

    print_signature()

    parser = HfArgumentParser((BaseModelArguments, BaseDataArguments, TrainingArguments))
    _, _, training_args = parser.parse_args_into_dataclasses()

    if training_args.experiment == "baseline":
        from src.baselines.data_argument import (
            BaselineDataArguments as KPADataArguments,
        )
        from src.baselines.datasets import (
            BaselineInferenceDataset as KPAInferenceDataset,
        )
        from src.baselines.datasets import BaselineTrainDataset as KPATrainDataset
        from src.baselines.model_argument import (
            BaselineModelArguments as KPAModelArguments,
        )
        from src.baselines.models import BaselineModel as KPAModel
    elif training_args.experiment == "qa":
        from src.question_answering.data_argument import (
            QADataArguments as KPADataArguments,
        )
        from src.question_answering.datasets import (
            QAInferenceDataset as KPAInferenceDataset,
        )
        from src.question_answering.datasets import QATrainDataset as KPATrainDataset
        from src.question_answering.model_argument import (
            QAModelArguments as KPAModelArguments,
        )
        from src.question_answering.models import QABertModel as KPAModel
    elif training_args.experiment == "triplet":
        from src.triplet.data_argument import TripletDataArguments as KPADataArguments
        from src.triplet.datasets import TripletInferenceDataset as KPAInferenceDataset
        from src.triplet.datasets import TripletTrainDataset as KPATrainDataset
        from src.triplet.model_argument import (
            TripletModelArguments as KPAModelArguments,
        )
        from src.triplet.models import TripletModel as KPAModel
    elif training_args.experiment == "mixed":
        from src.mixed.data_argument import MixedDataArguments as KPADataArguments
        from src.mixed.datasets import MixedInferenceDataset as KPAInferenceDataset
        from src.mixed.datasets import MixedTrainDataset as KPATrainDataset
        from src.mixed.model_argument import MixedModelArguments as KPAModelArguments
        from src.mixed.models import MixedModel as KPAModel
    elif training_args.experiment == "pseudolabel":
        from src.pseudo_label.data_argument import (
            PseudoLabelDataArguments as KPADataArguments,
        )
        from src.pseudo_label.datasets import (
            PseudoLabelInferenceDataset as KPAInferenceDataset,
        )
        from src.pseudo_label.datasets import PseudoLabelTrainDataset as KPATrainDataset
        from src.pseudo_label.model_argument import (
            PseudoLabelModelArguments as KPAModelArguments,
        )
        from src.pseudo_label.models import PseudoLabelModel as KPAModel
    else:
        raise NotImplementedError(
            f"Experiment {training_args.experiment} is not implemented yet. Must be in `baseline`, `qa`, `triplet`, `mixed` or `pseudolabel`"
        )

    parser = HfArgumentParser((KPAModelArguments, KPADataArguments, TrainingArguments))
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "model.yaml"), "w") as f:
        yaml.dump(vars(model_args), f, indent=2, default_flow_style=False)
    with open(os.path.join(training_args.output_dir, "data.yaml"), "w") as f:
        yaml.dump(vars(data_args), f, indent=2, default_flow_style=False)
    with open(os.path.join(training_args.output_dir, "training.yaml"), "w") as f:
        yaml.dump(vars(training_args), f, indent=2, default_flow_style=False)

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

    model = KPAModel(args=model_args)
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    logger.info(f"Number of parameters: {count_parameters(model)}")

    train_df, train_arg_df, train_kp_df, train_labels_df = get_data(gold_data_dir=data_args.directory, subset="train")
    val_df, val_arg_df, val_kp_df, val_labels_df = get_data(gold_data_dir=data_args.directory, subset="dev")
    test_df, test_arg_df, test_kp_df, test_labels_df = get_data(gold_data_dir=data_args.test_directory, subset="test")

    val_inf_df = prepare_inference_data(val_arg_df, val_kp_df)
    test_inf_df = prepare_inference_data(test_arg_df, test_kp_df)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    train_topic_word = word_len(train_arg_df["topic"].unique())
    train_argument_word = word_len(train_arg_df["argument"])
    train_key_point_word = word_len(train_kp_df["key_point"])
    train_topic_char = char_len(train_arg_df["topic"].unique())
    train_argument_char = char_len(train_arg_df["argument"])
    train_key_point_char = char_len(train_kp_df["key_point"])
    train_topic_token = token_len(train_arg_df["topic"].unique(), tokenizer)
    train_argument_token = token_len(train_arg_df["argument"], tokenizer)
    train_key_point_token = token_len(train_kp_df["key_point"], tokenizer)

    length_plot(train_topic_word, f"{data_args.directory}/train_topic_word.pdf")
    length_plot(train_argument_word, f"{data_args.directory}/train_argument_word.pdf")
    length_plot(train_key_point_word, f"{data_args.directory}/train_key_point_word.pdf")
    length_plot(train_topic_char, f"{data_args.directory}/train_topic_char.pdf")
    length_plot(train_argument_char, f"{data_args.directory}/train_argument_char.pdf")
    length_plot(train_key_point_char, f"{data_args.directory}/train_key_point_char.pdf")
    length_plot(train_topic_token, f"{data_args.directory}/train_topic_{tokenizer_type}.pdf")
    length_plot(train_argument_token, f"{data_args.directory}/train_argument_{tokenizer_type}.pdf")
    length_plot(train_key_point_token, f"{data_args.directory}/train_key_point_{tokenizer_type}.pdf")

    val_topic_word = word_len(val_arg_df["topic"].unique())
    val_argument_word = word_len(val_arg_df["argument"])
    val_key_point_word = word_len(val_kp_df["key_point"])
    val_topic_char = char_len(val_arg_df["topic"].unique())
    val_argument_char = char_len(val_arg_df["argument"])
    val_key_point_char = char_len(val_kp_df["key_point"])
    val_topic_token = token_len(val_arg_df["topic"].unique(), tokenizer)
    val_argument_token = token_len(val_arg_df["argument"], tokenizer)
    val_key_point_token = token_len(val_kp_df["key_point"], tokenizer)

    length_plot(val_topic_word, f"{data_args.directory}/val_topic_word.pdf")
    length_plot(val_argument_word, f"{data_args.directory}/val_argument_word.pdf")
    length_plot(val_key_point_word, f"{data_args.directory}/val_key_point_word.pdf")
    length_plot(val_topic_char, f"{data_args.directory}/val_topic_char.pdf")
    length_plot(val_argument_char, f"{data_args.directory}/val_argument_char.pdf")
    length_plot(val_key_point_char, f"{data_args.directory}/val_key_point_char.pdf")
    length_plot(val_topic_token, f"{data_args.directory}/val_topic_{tokenizer_type}.pdf")
    length_plot(val_argument_token, f"{data_args.directory}/val_argument_{tokenizer_type}.pdf")
    length_plot(val_key_point_token, f"{data_args.directory}/val_key_point_{tokenizer_type}.pdf")

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    train_dataset = KPATrainDataset(
        df=train_df,
        tokenizer=tokenizer,
        args=data_args,
    )

    val_dataset = KPAInferenceDataset(
        df=val_inf_df,
        arg_df=val_arg_df,
        labels_df=val_labels_df,
        tokenizer=tokenizer,
        args=data_args,
    )

    test_dataset = KPAInferenceDataset(
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
