from dataclasses import dataclass, field

from src.backbone.base_arguments import BaseModelArguments


@dataclass
class BaselineModelArguments(BaseModelArguments):

    """Arguments pertaining to which model/config/tokenizer we are going to
    fine- tune, or train from scratch."""

    loss_fct: str = field(default="constrastive", metadata={"help": "Loss function used for training siamese network."})
    margin: float = field(default=1.0, metadata={"help": "Margin distance value."})
