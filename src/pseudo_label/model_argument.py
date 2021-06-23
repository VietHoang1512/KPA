from dataclasses import dataclass, field

from src.backbone.base_arguments import BaseModelArguments


@dataclass
class ModelArguments(BaseModelArguments):

    """Arguments pertaining to which model/config/tokenizer we are going to
    fine- tune, or train from scratch."""

    margin: float = field(default=0.5, metadata={"help": "Margin distance value for cicle loss."})
