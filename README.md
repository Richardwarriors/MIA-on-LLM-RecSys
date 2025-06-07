# MIA-on-LLM-RecSys

This repository contains the code and experiments for studying **Membership Inference Attacks (MIA)** on **In-Context Learning-based Recommender Systems (ICL-RecSys)**.

## 🔍 Overview

We design and evaluate **four types of MIA attack strategies** against large language model (LLM)-based recommenders under the in-context learning paradigm:

1. **Inquiry Attack**  
2. **Hallucination Attack**  
3. **Similarity-based Attack**  
4. **Poisoning-based Attack**

These attacks are conducted in various settings to test how well LLMs such as LLaMA-3 preserve training-time privacy during recommendation tasks.

---

### Run an Example

You can run an MIA experiment with the following command:

```bash
python3 poisoning.py --models llama2 --datasets ml1m --num_seeds 500 --all_shots 1 --positions 'end'

### Results

### we give poisioning attack on llama2 on MovieLens-1M as an example.

| num_shots | num_poison | threshold | Adv    |
|-----------|------------|-----------|--------|
| 1         | 2          | 0.31      | 0.788  |
|           | 5          | 0.35      | 0.7913 |
|           | 8          | 0.34      | 0.7917 |
| 5         | 2          | 0.26      | 0.744  |
|           | 5          | 0.27      | 0.815  |
|           | 8          | 0.26      | 0.775  |
| 10        | 2          | 0.27      | 0.670  |
|           | 5          | 0.29      | 0.705  |
|           | 8          | 0.28      | 0.765  |

