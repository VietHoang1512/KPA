from dataclasses import dataclass, field
from typing import Optional

from qs_kpa.backbone.base_arguments import BaseDataArguments


@dataclass
class MixedDataArguments(BaseDataArguments):

    """Mixed Arguments pertaining to what data we are going to input our model for training and eval."""

    max_len: Optional[int] = field(default=36, metadata={"help": "max sequence length for topics and keypoints"})
    argument_max_len: Optional[int] = field(default=48, metadata={"help": "max sequence length for arguments"})
