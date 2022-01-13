from dataclasses import dataclass, field
from typing import Optional

from qs_kpa.backbone.base_arguments import BaseDataArguments


@dataclass
class PseudoLabelDataArguments(BaseDataArguments):

    """Pseudo Label Arguments pertaining to what data we are going to input our model for training and eval."""

    max_len: Optional[int] = field(default=36, metadata={"help": "max sequence length for topics and keypoints"})
    statement_max_len: Optional[int] = field(
        default=48, metadata={"help": "max sequence length for arguments/key points"}
    )
    max_pos: Optional[int] = field(default=5, metadata={"help": "maximum positive argument"})
    max_neg: Optional[int] = field(default=5, metadata={"help": "maximum negative argument"})
    max_unknown: Optional[int] = field(default=5, metadata={"help": "maximum unknown argument"})
