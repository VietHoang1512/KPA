import json
import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from src.utils import constants
from src.utils.data import evaluate_predictions
from src.utils.train_utils import AverageMeter, EarlyStopping

from .training_argument import TrainingArguments

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_inf_dataset: Dataset,
        val_inf_dataset: Dataset,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_inf_dataset = train_inf_dataset
        self.val_inf_dataset = val_inf_dataset
        self.optimizers = optimizers
        self.es = EarlyStopping(patience=self.args.early_stop, mode="max")
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )

        os.makedirs(self.args.output_dir, exist_ok=True)

    def get_train_dataloader(self) -> DataLoader:
        data_loader = DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers
        )
        return data_loader

    def get_val_dataloader(self, val_dataset: Dataset = None) -> DataLoader:
        val_dataset = val_dataset if val_dataset is not None else self.val_dataset
        data_loader = DataLoader(
            val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers
        )
        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to
        use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """Helper to get num of examples from a DataLoader, by accessing its
        Dataset."""
        return len(dataloader.dataset)

    def train(self, model_path: str = None):
        """Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()

        t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
        num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)
        model = self.model

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
            and os.path.isfile(os.path.join(model_path, "model.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
            logger.info("Loaded all previous model, optimizer, scheduler states")

        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))

        model.to(self.args.device)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Training batch size = %d", self.args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                global_step = 0
                logger.info("  Starting training from scratch")

        model.zero_grad()
        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
        for epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Training")
            epoch_iterator.set_postfix(lr=optimizer.param_groups[0]["lr"])
            total_train_loss = AverageMeter()

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                train_loss, n_train_samples = self._training_step(model, inputs)

                total_train_loss.update(train_loss, n_train_samples)
                epoch_iterator.set_postfix(TRAIN_LOSS=total_train_loss.avg)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):

                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
            logs = dict()

            if self.args.evaluate_during_training:
                logs["VAL_LOSS"] = self.evaluate(model, display_loss=True)
                logs["mAP_strict"], logs["mAP_relaxed"], prediction = self.evaluate(
                    model, val_dataset=self.val_inf_dataset
                )

                epoch_iterator.set_postfix(mAP_strict=logs["mAP_strict"], mAP_relaxed=logs["mAP_relaxed"])

                # Save model checkpoint
                output_dir = os.path.join(self.args.output_dir, "best_model")
                os.makedirs(output_dir, exist_ok=True)
                self.es(logs["mAP_strict"], model, optimizer, scheduler, output_dir)
                if self.es.is_best:
                    self._save_prediction(prediction=prediction, output_dir=output_dir)

            # Save model after each epoch
            output_dir = os.path.join(self.args.output_dir, f"{constants.PREFIX_CHECKPOINT_DIR}-{global_step}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            logs["TRAIN_LOSS"] = total_train_loss.avg
            if self.tb_writer:
                for k, v in logs.items():
                    self.tb_writer.add_scalar(k, v, epoch)
        if self.args.do_inference:
            # TODO: inference
            pass
        if self.tb_writer:
            self.tb_writer.close()

        if not self.args.evaluate_during_training:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", output_dir)
            self.es(0, model, optimizer, scheduler, output_dir)

    def _training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)

        loss = outputs[0]
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.item(), inputs[k].size(0)

    def evaluate(
        self,
        model: nn.Module,
        val_dataset: Optional[Dataset] = None,
        display_loss: Optional[bool] = False,
    ) -> Dict[str, float]:

        val_dataloader = self.get_val_dataloader(val_dataset)
        val_df = val_dataloader.dataset.df.copy()
        predictions = []
        epoch_iterator = tqdm(val_dataloader, desc="Evaluating")
        total_val_loss = AverageMeter()
        for inputs in epoch_iterator:
            val_loss, prob, n_val_samples = self._prediction_loop(model, inputs)
            total_val_loss.update(val_loss, n_val_samples)
            predictions.extend(prob)
            if display_loss:
                epoch_iterator.set_postfix(VAL_LOSS=total_val_loss.avg)

        val_df["label"] = predictions
        if not display_loss:
            return self.calculate_metric(val_df, val_dataloader.dataset.labels_df, val_dataloader.dataset.arg_df)
        else:
            return total_val_loss.avg

    def calculate_metric(self, val_df: pd.DataFrame, labels_df: pd.DataFrame, arg_df: pd.DataFrame):
        arg_df = arg_df[["arg_id", "topic", "stance"]].copy()

        predictions = {arg_id: dict() for arg_id in val_df["arg_id"].unique()}
        for _, row in val_df.iterrows():
            predictions[row["arg_id"]][row["keypoint_id"]] = row["label"]

        arg = []
        kp = []
        scores = []
        for arg_id, kps in predictions.items():
            best_kp = max(kps.items(), key=lambda x: x[1])
            arg.append(arg_id)
            kp.append(best_kp[0])
            scores.append(best_kp[1])

        predictions_df = pd.DataFrame({"arg_id": arg, "key_point_id": kp, "score": scores})
        # make sure each arg_id has a prediction
        predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

        # handle arguements with no matching key point
        predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
        predictions_df["score"] = predictions_df["score"].fillna(0)

        # merge each argument with the gold labels
        merged_df = pd.merge(predictions_df, labels_df.copy(), how="left", on=["arg_id", "key_point_id"])

        merged_df.loc[merged_df["key_point_id"] == "dummy_id", "label"] = 0
        merged_df["label_strict"] = merged_df["label"].fillna(0)
        merged_df["label_relaxed"] = merged_df["label"].fillna(1)
        return evaluate_predictions(merged_df), predictions

    def _save_prediction(self, prediction, output_dir):
        with open(os.path.join(output_dir, "prediction.p"), "w") as f:
            json.dump(prediction, f, indent=4)
        logger.info(f"Saved prediction to {output_dir}")

    def _prediction_loop(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:

        model.eval()
        with torch.no_grad():
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            outputs = model(**inputs)

        loss, prob = outputs
        prob = (
            prob.cpu()
            .detach()
            .numpy()
            .reshape(
                -1,
            )
            .tolist()
        )

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss.item(), prob, inputs[k].size(0)
