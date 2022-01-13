from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseDataArguments:

    """Base Arguments pertaining to what data we are going to input our model for training and eval."""

    directory: Optional[str] = field(default=None, metadata={"help": "The input data folder"})
    test_directory: Optional[str] = field(default=None, metadata={"help": "The test data folder"})


@dataclass
class BaseModelArguments:

    """Base Model Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    model_name: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "If training from scratch, pass a model type from https://huggingface.co/models "},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"},
    )

    tokenizer: Optional[str] = field(
        default="roberta-base", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    n_hiddens: int = field(
        default=4,
        metadata={
            "help": "Concatenate n_hiddens final layer of [CLS] token's representation. Use last pooler output if n_hiddens <0"
        },
    )

    drop_rate: float = field(default=0.1, metadata={"help": "Model dropout rate."})

    batch_norm: bool = field(default=False, metadata={"help": "Whether to use batch normalization or not."})

    stance_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding stance."})

    text_dim: int = field(default=256, metadata={"help": "Hidden representation dimension used for encoding text."})

    distance: str = field(
        default="cosine", metadata={"help": "Function that returns a distance between two embeddings."}
    )

    normalize: bool = field(default=False, metadata={"help": "Whether to normalize the embedding vectors or not."})

    stance_free: bool = field(default=False, metadata={"help": "Whether to use the stance information."})
