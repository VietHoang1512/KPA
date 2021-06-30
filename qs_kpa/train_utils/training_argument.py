import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from qs_kpa.train_utils.helpers import cached_property


@dataclass
class TrainingArguments:

    """TrainingArguments is the subset of the arguments we use in our example
    scripts."""

    experiment: str = field(metadata={"help": "Experiment type, `baseline`, `qa`, `triplet`, `mixed` or `pseudolabel`"})

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_inference: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False,
        metadata={
            "help": "Run evaluation during training at each logging step, this would be coupled with the number of early stopping rounds."
        },
    )

    logging_steps: int = field(default=0, metadata={"help": "Log every X updates steps."})

    train_batch_size: int = field(default=32, metadata={"help": "Batch size per GPU/CPU for training."})
    val_batch_size: int = field(default=32, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    early_stop: int = field(
        default=5,
        metadata={"help": "Number early stopping round for when the criterion met"},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: str = field(default=None, metadata={"help": "Tensorboard log dir."})

    seed: int = field(default=1512, metadata={"help": "Random seed for initialization"})

    no_cuda: bool = field(default=False, metadata={"help": "Whether to not use CUDA even when it is available or not."})

    num_workers: int = field(default=2, metadata={"help": "Number of workers for data loading."})

    @property
    def device(self) -> torch.device:
        """
        The device used by this process.
        """
        return self._setup_devices

    @cached_property
    def _setup_devices(self):
        if self.no_cuda:
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Sanitized serialization to use with TensorBoard’s hparams."""
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        valid_types.append(torch.Tensor)
        return {k: v if isinstance(v, tuple(valid_types)) else str(v) for k, v in d.items()}
