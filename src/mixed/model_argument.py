from dataclasses import dataclass, field

from src.train_utils.base_arguments import BaseModelArguments


@dataclass
class ModelArguments(BaseModelArguments):
    stance_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding stance."})
    text_dim: int = field(default=32, metadata={"help": "Hidden representation dimension used for encoding text."})
    loss_fct: str = field(default="constrastive", metadata={"help": "Loss function used for constrastive network."})
    sample_selection: str = field(default="all", metadata={"help": "Online triplet selection strategy."})
    distance: str = field(
        default="euclidean", metadata={"help": "Function that returns a distance between two emeddings."}
    )
    pair_margin: float = field(default=0.5, metadata={"help": "Margin distance value for pair loss."})
    triplet_margin: float = field(default=0.5, metadata={"help": "Margin distance value for triplet loss."})
