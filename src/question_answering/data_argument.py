from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:

    """Arguments pertaining to what data we are going to input our model for training and eval."""

    directory: Optional[str] = field(default=None, metadata={"help": "The input data folder"})
    max_topic_length: Optional[int] = field(default=36, metadata={"help": "max sequence length for topics"})
    max_statement_length: Optional[int] = field(
        default=48, metadata={"help": "max sequence length for arguments/key points"}
    )
