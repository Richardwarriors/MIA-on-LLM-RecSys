import numpy as np
from copy import deepcopy
import os
from transformers import GPT2Tokenizer

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")


def random_sampling(sentences, num):
    if num > len(sentences):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(sentences)}"
    idxs = np.random.choice(len(sentences), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    return deepcopy(selected_sentences)