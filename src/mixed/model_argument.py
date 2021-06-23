from dataclasses import dataclass, field

from src.backbone.base_arguments import BaseModelArguments


@dataclass
class ModelArguments(BaseModelArguments):
    loss_fct: str = field(default="constrastive", metadata={"help": "Loss function used for constrastive network."})
    sample_selection: str = field(default="all", metadata={"help": "Online triplet selection strategy."})
    pair_margin: float = field(default=0.5, metadata={"help": "Margin distance value for pair loss."})
    triplet_margin: float = field(default=0.5, metadata={"help": "Margin distance value for triplet loss."})
