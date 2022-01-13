import os
import random

import numpy as np
import torch

from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


def seed_everything(seed: int) -> None:
    """
    Seed for reproceducing.

    Args:
        seed (int): seed number
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    def __init__(self):
        """Average Meter class for Pytorch experiments."""
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 0):
        """Early stopping when the criterion met."""
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_best = True
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, optimizer, scheduler, output_dir):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, output_dir)
            return True
        elif score < self.best_score + self.delta:
            self.is_best = False
            self.counter += 1
            print("Validation score: {} which is not an improvement from {}".format(score, self.best_score))
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, output_dir)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, optimizer, scheduler, output_dir):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model to {}!".format(
                    self.val_score, epoch_score, output_dir
                )
            )
            logger.info("Saving best optimizer and scheduler states to %s", output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        self.val_score = epoch_score


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.
    From tensorflow_datasets
    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached
