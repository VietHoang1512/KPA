import torch
from transformers import AutoTokenizer

from src.baselines.model_argument import BaselineModelArguments
from src.baselines.models import BaselineBertModel

if __name__ == "__main__":

    BERT_MODEL = "albert-base-v2"
    BATCH_SIZE = 8
    MAXLEN = 16

    model_argument = BaselineModelArguments
    model_argument.model_name_or_path = BERT_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = BaselineBertModel(args=model_argument)

    topics = [
        "Argument mining  is a young and gradually maturing research area within computational linguistics"
    ] * BATCH_SIZE
    arguments = ["Argument mining gives rise to various practical applications of great importance"] * BATCH_SIZE
    key_points = [
        "It provides methods that can find and visualize the main pro and con arguments in a text corpus"
    ] * BATCH_SIZE
    stances = torch.tensor([1] * BATCH_SIZE, dtype=torch.float).view(-1, 1)
    labels = torch.tensor([1] * BATCH_SIZE, dtype=torch.float).view(-1, 1)

    topics_encoded = tokenizer.batch_encode_plus(
        topics, max_length=MAXLEN, padding="max_length", return_token_type_ids=True, truncation=True
    )
    topics_input_ids = torch.tensor(topics_encoded["input_ids"], dtype=torch.long)
    topics_attention_mask = torch.tensor(topics_encoded["attention_mask"], dtype=torch.long)
    topics_token_type_ids = torch.tensor(topics_encoded["token_type_ids"], dtype=torch.long)

    arguments_encoded = tokenizer.batch_encode_plus(
        arguments, max_length=MAXLEN, padding="max_length", return_token_type_ids=True, truncation=True
    )
    arguments_input_ids = torch.tensor(arguments_encoded["input_ids"], dtype=torch.long)
    arguments_attention_mask = torch.tensor(arguments_encoded["attention_mask"], dtype=torch.long)
    arguments_token_type_ids = torch.tensor(arguments_encoded["token_type_ids"], dtype=torch.long)

    key_points_encoded = tokenizer.batch_encode_plus(
        key_points, max_length=MAXLEN, padding="max_length", return_token_type_ids=True, truncation=True
    )
    key_points_input_ids = torch.tensor(key_points_encoded["input_ids"], dtype=torch.long)
    key_points_attention_mask = torch.tensor(key_points_encoded["attention_mask"], dtype=torch.long)
    key_points_token_type_ids = torch.tensor(key_points_encoded["token_type_ids"], dtype=torch.long)

    model.eval()
    prob = model(
        topics_input_ids,
        topics_attention_mask,
        topics_token_type_ids,
        key_points_input_ids,
        key_points_attention_mask,
        key_points_token_type_ids,
        arguments_input_ids,
        arguments_attention_mask,
        arguments_token_type_ids,
        stances,
        labels,
    )
    print("DONE!")
