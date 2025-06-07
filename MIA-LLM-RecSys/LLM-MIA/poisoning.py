import sys
import os
import re
import argparse
from datetime import datetime
import numpy as np
import pickle
import random
import requests
import json
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_dataset,
    random_sampling,
)

def main(models, datasets, num_seeds, positions, all_shots):
    default_params = {
        "conditioned_on_correct_classes": True,
    }

    all_params = []
    for model in models:
        for dataset in datasets:
            for position in positions:
                for num_shots in all_shots:
                    for seed in range(num_seeds):
                        p = deepcopy(default_params)
                        p["model"] = model
                        p["dataset"] = dataset
                        p["seed"] = seed
                        p["num_shots"] = num_shots
                        p['position'] = position
                        p["expr_name"] = f"{p['dataset']}_{p['model']}_subsample_seed{p['seed']}"
                        all_params.append(p)


    all_member_list = []
    all_nonmember_list = []

    semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for param_index, params in enumerate(all_params):
        prompt_subset = prepare_data(params)

        member_pool = prompt_subset[:len(prompt_subset)//2]
        nonmember_pool = prompt_subset[len(prompt_subset)//2:]

        member_sentences = random.sample(member_pool, params['num_shots'])

        target_sentence = member_sentences[-1] if params['position'] == 'end' else member_sentences[0]
        nonmember_sentences = random.sample(nonmember_pool, params['num_shots'])
        nontarget_sentence = nonmember_sentences[0]

        required_for_mem = posion(params, member_sentences, target_sentence,semantic_model)
        if required_for_mem is None:
            continue

        print(100 * '-')
        required_for_nonmem = posion(params, member_sentences, nontarget_sentence,semantic_model)
        if required_for_nonmem is None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)

        save_path = f"../results/poision/poision_5/{params['model']}/{params['position']}/{params['num_shots']}_shots/"
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'member.pkl'), "wb") as file:
            pickle.dump(all_member_list, file)
        with open(os.path.join(save_path, 'nonmember.pkl'), "wb") as file:
            pickle.dump(all_nonmember_list, file)

def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    prompted_sentences = load_dataset(params)
    np.random.seed(params["seed"])
    return random_sampling(prompted_sentences, 1000)

def posion(params, member_sentences, test_sentence,semantic_model):
    user_match = re.search(r'The user with id (\d+)', test_sentence)
    rec_match = re.search(r'watched (.+?) and based on', test_sentence)
    if user_match and rec_match:
        user_id = int(user_match.group(1))
        interaction_list = rec_match.group(1)
    else:
        return None
    query_sentence = f"Please recommend top-10 movies with descending order for {user_id}? Only give movie name with a list and not give any description."
    input_to_model = construct_prompt_cut(params, member_sentences, query_sentence)
    print(f"input_to_model: {input_to_model}")

    base_sentence = continue_generate(input_to_model, query_sentence, params["model"])
    print(f"return_sentence: {base_sentence}")

    base_movie_list = re.findall(r'\d+\.\s+(.*)', base_sentence)
    print(f"movie_list: {base_movie_list}")
    if len(base_movie_list) == 0:
        return None
    print(f"interaction_list: ", interaction_list)
    print(type(interaction_list))
    interaction_list = [movie.strip() for movie in interaction_list.split('|')]
    interaction_embeddings = []
    print(f"interaction_list length: {len(interaction_list)}")
    for movie in list(interaction_list):
        embedding = semantic_model.encode(movie, convert_to_tensor=True)
        interaction_embeddings.append(embedding)
    print(f"interaction_embedding length: ", len(interaction_embeddings))
    base_semantic_sim = 0
    for i in range(len(base_movie_list)):
        temp = 0
        embedding_movie = semantic_model.encode(base_movie_list[i], convert_to_tensor=True)
        for j in range(len(interaction_embeddings)):
            temp += util.pytorch_cos_sim(embedding_movie, interaction_embeddings[j]).item()
        base_semantic_sim += (temp / len(interaction_embeddings))
    base_semantic_sim = base_semantic_sim / len(base_movie_list)

    posion_number = 5
    poison_input_to_model = construct_prompt_poison(params, member_sentences, test_sentence, query_sentence, semantic_model,posion_number)

    poison_sentence = continue_generate(poison_input_to_model, query_sentence, params["model"])
    print(f"poison_sentence: {poison_sentence}")

    poison_movie_list = re.findall(r'\d+\.\s+(.*)', poison_sentence)
    print(f"poison_movie_list: {poison_movie_list}")

    if poison_movie_list == []:
        return None
    poision_change_semantic = 0
    for i in range(len(base_movie_list)):
        temp = 0
        embedding_movie = semantic_model.encode(base_movie_list[i], convert_to_tensor=True)
        for j in range(len(poison_movie_list)):
            poision_embedding = semantic_model.encode(poison_movie_list[j], convert_to_tensor=True)
            temp += util.pytorch_cos_sim(embedding_movie, poision_embedding).item()
        poision_change_semantic += (temp / len(poison_movie_list))
    poision_change_semantic = poision_change_semantic / len(base_movie_list)

    return poision_change_semantic

def continue_generate(prompt_setup, prompt_question, model, max_token = 1000, temperature=0.3):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_setup},
            {"role": "user", "content": prompt_question}
        ],
        "max_tokens": max_token,
        "temperature": temperature,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("message", {}).get("content", "").strip().lower()
        return raw_output
    except requests.RequestException as e:
        print(f"[Error] Request to Ollama chat API failed: {e}")
        return -1

def construct_prompt_cut(params, train_sentences, query_sentence):
    prompt = params.get("prompt_prefix", "")
    print(f"type of train_sentences: {type(train_sentences)}")
    prompt += "\n".join(train_sentences) + "\n\n" + query_sentence
    return prompt

def construct_prompt_poison(params, train_sentences, test_sentence, query_sentence, semantic_model,poision_number):
    prompt = params.get("prompt_prefix", "")

    user_match = re.search(r'The user with id (\d+)', test_sentence)
    interaction_match = re.search(r'watched (.+?) and based on', test_sentence)

    if not (user_match and interaction_match):
        return None

    user_id = int(user_match.group(1))
    interaction_list = interaction_match.group(1).split('|')

    file_path = "../data/IMDB/title.basics.tsv"
    IMDB = pd.read_csv(file_path, sep="\t", low_memory=False)
    IMDB_dataset_primary = list(IMDB['primaryTitle'].dropna())
    IMDB_sampled = random.sample(IMDB_dataset_primary, 1000)

    IMDB_sampled_embeddings = semantic_model.encode(IMDB_sampled, convert_to_tensor=True)

    selected_movies = random.sample(interaction_list, poision_number)

    for movie in selected_movies:
        movie_embedding = semantic_model.encode(movie, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(movie_embedding, IMDB_sampled_embeddings)[0]

        best_idx = cos_scores.argmax().item()
        replacement_movie = IMDB_sampled[best_idx]

        movie_idx = interaction_list.index(movie)
        interaction_list[movie_idx] = replacement_movie

        print(f"Replaced '{movie}' ➝ '{replacement_movie}'")

    new_watched_str = '|'.join(interaction_list)

    poisoned_prompt = (
        f"The user with id {user_id} watched {new_watched_str} "
        f"and based on his or her watched history, the top 10 recommended item with descending order is in the following: "
    )

    for i in range(len(train_sentences)):
        if str(user_id) in train_sentences[i]:
            train_sentences[i] = poisoned_prompt

    prompt += "\n".join(train_sentences) + "\n\n" + query_sentence
    return prompt


def convert_to_list(items, is_int=False):
    return [int(s.strip()) if is_int else s.strip() for s in items.split(",")]

if __name__ == "__main__":
    random.seed(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--num_seeds", type=int, required=True)
    parser.add_argument("--all_shots", required=True)
    parser.add_argument("--positions", required=True)

    args = vars(parser.parse_args())
    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    args["positions"] = convert_to_list(args["positions"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)

    main(**args)
