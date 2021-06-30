<div align="center">

# 🏅[Quantitative Summarization – Key Point Analysis](https://competitions.codalab.org/competitions/31166)🏅

[![CircleCI](https://circleci.com/gh/VietHoang1512/KPA.svg?style=svg&circle-token=a196c015fd323b139ee617a2ebd36b9055dee3a2)](https://circleci.com/gh/VietHoang1512/KPA/tree/main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://colab.research.google.com/drive/1RNZsW30ulRs5Avkwe8Jqfc8zRbhpmUbD?usp=sharing">
<img src="https://img.shields.io/badge/stable-ff9f01?logo=google-colab&logoColor=white">
</a>
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/776410d9c5ea4290b0301d5f70bec9b5)](https://www.codacy.com/gh/VietHoang1512/KPA/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=VietHoang1512/KPA&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/viethoang1512/kpa/badge?s=805044f88408096519ce8ab36564bb8b98e8e9ba)](https://www.codefactor.io/repository/github/viethoang1512/kpa)

</div>


## 💪Installation

```bash
pip install keypoint-analysis
```

## TODO:

### Pair-wise keypoint-argument
Given a pair of key point and argument (along with their supported topic & stance) and the matching score. Similar pairs with label 1 are pulled together, or pushed away otherwise.

#### Model

| Model               | BERT/ConvBERT               | DistilBERT         | ALBERT             | XLNet            | RoBERTa                | ELECTRA            | BART            |
| ------------------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Siamese Baseline            | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| Siamese Question Answering-like              | ✔️ | ✔️ | ✔️ |  | ✔️ | ✔️ | ✔️ |
| Custom loss Baseline             | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |

#### Loss

- [x] Constrastive
- [x] Online Constrastive
- [x] Triplet
- [x] Online Triplet (Hard negative/positive mining)

#### Distance

- [x] Euclidean
- [x] Cosine
- [x] Manhattan

### Pseudo-label

Group the arguments by their key point and consider the order of that key point within the topic as their labels (see [pseudo_label](src/pseudo_label)). We can now utilize available pytorch metrics learning distance, losses, miners or reducers from this great [open-source](https://github.com/KevinMusgrave/pytorch-metric-learning) in the main training workflow. This is also our best approach (single-model) so far.

![Model architecture](assets/model.png "Model architecture")

### Utils

- [x] K-folds
- [x] Full-flow

## Contributors

- Phan Viet Hoang
- Nguyen Duc Long