import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """
    Seed for reproceducing
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
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
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
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, output_dir)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
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
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        self.val_score = epoch_score


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)
