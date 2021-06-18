from dataclasses import dataclass, field

from src.train_utils.base_arguments import BaseModelArguments


@dataclass
class ModelArguments(BaseModelArguments):

    """Arguments pertaining to which model/config/tokenizer we are going to
    fine- tune, or train from scratch."""

    stance_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding stance."})
    text_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding text."})
    distance: str = field(
        default="cosine", metadata={"help": "Function that returns a distance between two emeddings."}
    )
    margin: float = field(default=0.5, metadata={"help": "Margin distance value for cicle loss."})
