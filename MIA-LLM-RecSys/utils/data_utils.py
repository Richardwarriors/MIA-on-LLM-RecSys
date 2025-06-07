import pandas as pd
import json
import pickle
import numpy as np
import os
from copy import deepcopy


def load_ml1m():
    prompt_sentences = []
    with open(f"../processed_ml-1m/prompt_ml1m.txt", "r") as prompt_data:
        for line in prompt_data:
            prompt_sentences.append(line)
    return prompt_sentences


def load_dataset(params):

    if params["dataset"] == "ml1m":
        prompt_sentences = load_ml1m()
        params[
            "prompt_prefix"
        ] = "Pretend you are a movie recommender system. And You task is to recommend top 10_shots movies which user will watch and recommended movies should not be user watched movies.\n\n"
        params["task_format"] = "recommendation"
        params["num_tokens_to_predict"] = 1
    else:
        raise NotImplementedError
    return prompt_sentences

if __name__ == "__main__":
    params = {}
    params["dataset"] = "ml1m"
    load_dataset(params)
    print(params)
