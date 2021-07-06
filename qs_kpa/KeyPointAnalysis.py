import os
from typing import List, Optional, Tuple, Union

import gdown
import numpy as np
import torch
from tqdm.auto import trange
from transformers import AutoTokenizer

from qs_kpa.pseudo_label.model_argument import PseudoLabelModelArguments
from qs_kpa.pseudo_label.models import PseudoLabelModel
from qs_kpa.utils.logging import custom_logger

URL = "https://drive.google.com/uc?id=1--TBb-i41BAtaPUWX5rz4EAXuelHiLTg"
MODEL_WEIGHT = "kpa_model_256.pt"
MAX_LEN = 30
STATEMENT_MAX_LEN = 50

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = custom_logger()


class KeyPointAnalysis(object):
    """
    Loads a KPA model, that can be used to map statement to embeddings.
    """

    def __init__(
        self,
        from_pretrained: Optional[bool] = True,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):

        if cache_folder is None:
            try:
                from torch.hub import _get_torch_home

                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(
                    os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
                )

            cache_folder = os.path.join(torch_cache_home, "key_point_analysis")

        model_args = PseudoLabelModelArguments()
        self.model = PseudoLabelModel(args=model_args)

        if from_pretrained:
            loaded = False
            if model_path is not None:
                if os.path.exists(model_path):
                    try:
                        self._load_model(model_path=model_path, model=self.model)
                        loaded = True
                    except Exception as e:
                        logger.warning(e)

            if not loaded:
                os.makedirs(cache_folder, exist_ok=True)
                model_path = os.path.join(cache_folder, MODEL_WEIGHT)
                try:
                    self._load_model(model_path=model_path, model=self.model)
                except Exception as e:
                    logger.warning(e)
                    self._download_and_cache(model_path)
                    self._load_model(model_path=model_path, model=self.model)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"Device: {self.device} \n"
        s += f"Backbone configuration:\n{self.model}"
        return s

    @classmethod
    def _download_and_cache(self, model_path: str) -> None:
        gdown.download(URL, model_path, quiet=False)
        logger.info("Pretrained model weights downloaded from cloud storage")

    @classmethod
    def _load_model(self, model_path: str, model: PseudoLabelModel) -> None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        logger.info(f"Loaded model from {model_path}")

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        examples: Union[List[Tuple[str, str, int]], Tuple[str, str, int]],
        batch_size: int = 8,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_tuple = False
        if isinstance(examples, tuple):  # Cast an individual tuple to a list with length 1
            examples = [examples]
            input_was_tuple = True

        examples = np.array(examples)

        all_embeddings = []

        for start_index in trange(0, len(examples), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = examples[start_index : start_index + batch_size]
            batch = self._convert_examples_to_features(batch)

            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_embeddings(**batch)

            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_tuple:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _convert_examples_to_features(self, examples: List[Tuple[str, str, int]]):
        topic = examples[:, 0]
        statement = examples[:, 1]
        stance = np.array(examples[:, 2], dtype=float)

        topic_inputs = self.tokenizer.batch_encode_plus(
            topic,
            max_length=MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        statement_inputs = self.tokenizer.batch_encode_plus(
            statement,
            max_length=STATEMENT_MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        batch = {
            "topic_input_ids": torch.tensor(topic_inputs["input_ids"], dtype=torch.long),
            "topic_attention_mask": torch.tensor(topic_inputs["attention_mask"], dtype=torch.long),
            "topic_token_type_ids": torch.tensor(topic_inputs["token_type_ids"], dtype=torch.long),
            "statement_input_ids": torch.tensor(statement_inputs["input_ids"], dtype=torch.long),
            "statement_attention_mask": torch.tensor(statement_inputs["attention_mask"], dtype=torch.long),
            "statement_token_type_ids": torch.tensor(statement_inputs["token_type_ids"], dtype=torch.long),
            "stance": torch.tensor(stance, dtype=torch.float).view(-1, 1),
        }

        return batch
