from dataclasses import dataclass, field

from qs_kpa.backbone.base_arguments import BaseModelArguments


@dataclass
class TripletModelArguments(BaseModelArguments):

    """Triplet Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    margin: float = field(default=0.5, metadata={"help": "Margin distance value for cicle loss."})
    sample_selection: str = field(default="all", metadata={"help": "Online triplet selection strategy."})
