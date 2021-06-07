from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:

    """Arguments pertaining to which model/config/tokenizer we are going to
    fine- tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from https://huggingface.co/models "},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"},
    )
    tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    n_hiddens: int = field(
        default=4, metadata={"help": "Concatenate n_hiddens final layer of [CLS] token's representation."}
    )
    stance_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding stance."})
    text_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding text."})
    drop_rate: float = field(default=0.1, metadata={"help": "Model dropout rate."})
    margin: float = field(default=1.0, metadata={"help": "Margin distance value."})
