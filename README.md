<div align="center">

# 🏅[Quantitative Summarization – Key Point Analysis](https://competitions.codalab.org/competitions/31166)🏅

[![CircleCI](https://circleci.com/gh/VietHoang1512/KPA.svg?style=svg&circle-token=a196c015fd323b139ee617a2ebd36b9055dee3a2)](https://circleci.com/gh/VietHoang1512/KPA/tree/main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RNZsW30ulRs5Avkwe8Jqfc8zRbhpmUbD?usp=sharing)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/776410d9c5ea4290b0301d5f70bec9b5)](https://www.codacy.com/gh/VietHoang1512/KPA/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=VietHoang1512/KPA&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/viethoang1512/kpa/badge?s=805044f88408096519ce8ab36564bb8b98e8e9ba)](https://www.codefactor.io/repository/github/viethoang1512/kpa)

</div>



## Keypoint Analysis 

This library is based on the Transformers library by HuggingFace. **Keypoint Analysis** quickly embedds the statements with provided supported topic and the stances toward that topic.

### What's New

#### July 1, 2021

- First release of [keypoint-analysis](https://pypi.org/project/keypoint-analysis/) python package

### Installation

```bash
pip install keypoint-analysis
```

### Quick example

```python
# Import needed libraries
from qs_kpa import KeyPointAnalysis

# Create a KeyPointAnalysis model
encoder = KeyPointAnalysis()

# Model configuration
print(encoder)

# Preparing data (a tuplet of (topic, statement, stance) or a list of tuple)
inputs = [
    (
        "Assisted suicide should be a criminal offence",
        "a cure or treatment may be discovered shortly after having ended someone's life unnecessarily.",
        1,
    ),
    (
        "Assisted suicide should be a criminal offence",
        "Assisted suicide should not be allowed because many times people can still get better",
        1,
    ),
    ("Assisted suicide should be a criminal offence", "Assisted suicide is akin to killing someone", 1),
]

# Go and embedd everything
output = encoder.encode(inputs, convert_to_numpy=True)
```

### Detailed training

Given a pair of key point and argument (along with their supported topic & stance) and the matching score. Similar pairs with label 1 are pulled together, or pushed away otherwise.

#### Model

| Model               | BERT/ConvBERT               | DistilBERT         | ALBERT             | XLNet            | RoBERTa                | ELECTRA            | BART            |
| ------------------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Siamese Baseline            | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| Siamese Question Answering-like              | ✔️ | ✔️ | ✔️ |  | ✔️ | ✔️ | ✔️ |
| Custom loss Baseline             | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |

#### Loss

- Constrastive
- Online Constrastive
- Triplet
- Online Triplet (Hard negative/positive mining)

#### Distance

- Euclidean
- Cosine
- Manhattan

#### Utils

- K-folds
- Full-flow

### Pseudo-label

Group the arguments by their key point and consider the order of that key point within the topic as their labels (see [pseudo_label](qs_kpa/pseudo_label)). We can now utilize available pytorch metrics learning distance, losses, miners or reducers from this great [open-source](https://github.com/KevinMusgrave/pytorch-metric-learning) in the main training workflow. This is also our best approach (single-model) so far.

![Model architecture](https://user-images.githubusercontent.com/52401767/124059293-0ec81100-da55-11eb-94a4-cf9914479a78.png)

### Training data

**ArgKP** dataset ([Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf))

### Contributors

- Phan Viet Hoang
- Nguyen Duc Long

### BibTeX

```bibtex
@misc{hoang2021qskpa,
  author = {Phan, V.H. & Nguyen, D.L.},
  title = {Keypoint Analysis},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/VietHoang1512/KPA}}
}
```