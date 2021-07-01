from dataclasses import dataclass, field

from qs_kpa.backbone.base_arguments import BaseModelArguments


@dataclass
class PseudoLabelModelArguments(BaseModelArguments):

    """Pseudo Label Arguments pertaining to which model/config/tokenizer we are going to fine- tune, or train from scratch."""

    margin: float = field(default=0.3, metadata={"help": "Margin distance value for cicle loss."})
