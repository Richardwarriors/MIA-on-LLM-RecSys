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
python3 semantic.py --models llama3 --datasets ml1m --num_seeds 10 --all_shots 1 --positions 'end'
