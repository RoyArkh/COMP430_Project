import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from datasets import load_dataset
from tqdm import tqdm
from heapq import nlargest
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import argparse
# Define functions

def get_all_texts(dataset_split):
    texts = dataset_split['text']
    labels = dataset_split['label']
    return texts, labels

import torch  
import numpy as np  
from heapq import nlargest  

def generate_neighbours(text, tokenizer, model, token_dropout, num_neighbors=50):  
    # Tokenize input text with padding and truncation, adding special tokens  
    text_tokenized = tokenizer(  
        text,  
        padding=True,  
        truncation=True,  
        max_length=512,  
        return_tensors='pt',  
        add_special_tokens=True  # To include [CLS] and [SEP]  
    ).input_ids.to('cuda')  # Ensure to send to the appropriate device  

    candidate_scores = {}  
    replacements = {}  

    # Iterate over each token, skipping [CLS] at index 0 and [SEP] at last index  
    for target_token_index in range(1, len(text_tokenized[0]) - 1):  
        target_token = text_tokenized[0, target_token_index]  

        # Get the embeddings for this input text  
        embeds = model.bert.embeddings(text_tokenized)  

        # Replace the target token's embedding  
        embeds = torch.cat((  
            embeds[:, :target_token_index, :],  
            token_dropout(embeds[:, target_token_index, :]).unsqueeze(dim=0),  
            embeds[:, target_token_index + 1:, :]  
        ), dim=1)  

        # Predict probabilities over the token vocabulary  
        token_probs = torch.softmax(model(inputs_embeds=embeds).logits, dim=2)  

        # Get the original token's probability  
        original_prob = token_probs[0, target_token_index, target_token].item()  

        # Get the top 10 candidates for replacement  
        top_probabilities, top_candidates = torch.topk(token_probs[:, target_token_index, :], 10, dim=1)  

        # Calculate scores and build the candidate replacements  
        for cand, prob in zip(top_candidates[0], top_probabilities[0]):  
            if cand != target_token:  
                # Calculate the replacement boost score  
                score = prob.item() / (1 - original_prob + 1e-8)  # Avoid division by zero  

                # Store the text and score for each candidate  
                alt = torch.cat((  
                    text_tokenized[:, :target_token_index],  
                    torch.LongTensor([cand.item()]).unsqueeze(0).to('cuda'),  # Move candidate token to the right device  
                    text_tokenized[:, target_token_index + 1:]  
                ), dim=1)  

                alt_text = tokenizer.batch_decode(alt)[0]  # Decode the altered tokenized text  
                candidate_scores[alt_text] = score  
                replacements[(target_token_index, cand.item())] = score  

    # Get the top n neighbors based on scores  
    highest_scored_texts = nlargest(num_neighbors, candidate_scores, key=candidate_scores.get)  

    return highest_scored_texts


def get_logprob(text, tokenizer, model):
    # Tokenize with [CLS] and [SEP] tokens
    text_tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
        add_special_tokens=True  # Ensures [CLS] and [SEP] are added
    ).input_ids.to('cuda')

    logprob = -model(text_tokenized, labels=text_tokenized).loss.item()
    return logprob



def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)

    if np.any(fpr_list <= 0.05):
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    else:
        idx = (np.abs(fpr_list - 0.05)).argmin()
        tpr05 = tpr_list[idx]
        print("Warning: No exact FPR of 0.05 found. Using closest value.")

    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]] if np.any(tpr_list >= 0.95) else np.nan
    return auroc, fpr95, tpr05

def save_neighbors_to_csv(neighbors_data, filename="Neighbourhood/NeighboursSaved/neighbours_xsum_32_N50.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_text"] + [f"neighbor_{i+1}" for i in range(50)]) # Dynamic header
        for original_text, neighbors in neighbors_data.items():
            writer.writerow([original_text] + neighbors)

def load_neighbors_from_csv(filename="Neighbourhood/NeighboursSaved/neighbours_xsum_32_N50.csv"):
    neighbors_data = {}
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                original_text = row[0]
                neighbors = row[1:]
                neighbors_data[original_text] = neighbors
    except FileNotFoundError:
        print("Neighbors CSV file not found. Generating neighbors...")
        return None
    return neighbors_data


# ########################################################### For Other Models M####################################################

parser = argparse.ArgumentParser()
parser.add_argument('--modelToAttack', type=str, default='mia-llm/MobileLLM-125M-wikitext2raw-hosein', 
                    choices=["mia-llm/pythia-70m-wikitext2raw-roya",
                            "mia-llm/pythia-160m-wikitext2raw-roya", "mia-llm/pythia-410m-wikitext2raw-roya", "mia-llm/pythia-1b-wikitext2raw-roya", "mia-llm/pythia-1.4b-wikitext2raw-roya",
                            "mia-llm/gpt-neo-125m-wikitext2raw-roya", "mia-llm/gpt-neo-1.3B-wikitext2raw-roya",
                            "mia-llm/MobileLLM-125M-wikitext2raw-hosein", "mia-llm/MobileLLM-350M-wikitext2raw-roya", "mia-llm/MobileLLM-600M-wikitext2raw-roya", "mia-llm/MobileLLM-1B-wikitext2raw-roya", "mia-llm/MobileLLM-1.5B-wikitext2raw-roya",
                            "mia-llm/pythia-160m-AGnews-roya",
                            "mia-llm/pythia-410m-AGnews-roya",
                            "mia-llm/pythia-1b-AGnews-roya",
                            "mia-llm/pythia-1.4b-wikitext2raw-roya",
                            "mia-llm/gpt-neo-125m-AGnews-roya", "mia-llm/gpt-neo-1.3B-AGnews-roya",
                            "mia-llm/MobileLLM-350M-AGnews-roya", "mia-llm/MobileLLM-600M-AGnews-roya", "mia-llm/MobileLLM-1B-AGnews-roya", "mia-llm/MobileLLM-1.5B-AGnews-roya",
                            "mia-llm/pythia-70m-xsum-roya", "mia-llm/pythia-160m-xsum-roya", "mia-llm/pythia-410m-xsum-roya", "mia-llm/pythia-1b-xsum-roya",
                            "mia-llm/gpt-neo-125m-xsum-roya", "mia-llm/gpt-neo-1.3B-xsum-roya"])
parser.add_argument('--dataset', type=str, default='h0ssn/Agnews_MIA_benchmark_RH', 
                    choices=["h0ssn/Agnews_MIA_benchmark_RH",
                    "h0ssn/Xsum_MIA_benchmark_RH"])


args = parser.parse_args()

attack_model_name = args.modelToAttack
#print(attack_model_name)

if "mobilellm" in attack_model_name.lower():
    print("trying to get mobilellm")
    attack_tokenizer = AutoTokenizer.from_pretrained(attack_model_name, trust_remote_code=True, use_fast=False)
    attack_tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        }
    )
    attack_model = AutoModelForCausalLM.from_pretrained(attack_model_name, trust_remote_code=True).to('cuda')
else:
    attack_tokenizer = AutoTokenizer.from_pretrained(attack_model_name, trust_remote_code=True)
    attack_model = AutoModelForCausalLM.from_pretrained(attack_model_name, trust_remote_code=True).to('cuda')

attack_model_name = str(args.modelToAttack[8:])

# Load model and dataset
# attack_tokenizer = AutoTokenizer.from_pretrained('mia-llm/MobileLLM-350M-wikitext2raw-roya', trust_remote_code=True, use_fast=False)
# attack_tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>", "pad_token": "<pad>"})
# attack_model = AutoModelForCausalLM.from_pretrained('mia-llm/MobileLLM-350M-wikitext2raw-roya', trust_remote_code=True).to('cuda')

search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
search_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to('cuda')

dataset_name = args.dataset
dataset = load_dataset(args.dataset, split='dataset_32')
texts, labels = get_all_texts(dataset)

# Initialize components
batch_size = 1200
token_dropout = torch.nn.Dropout(p=0.7)
all_attack_scores = []

print(f"going to run attack on {attack_model_name} with {dataset_name}")

# Load or Generate Neighbors
neighbors_data = load_neighbors_from_csv()
if neighbors_data is None:
    neighbors_data = {}
    total_generate_neighbours_time = 0
    for text in tqdm(texts, desc="Generating Neighbors"): # tqdm for generating neighbors
        with torch.no_grad():
            start_time = time.time()
            neighbors = generate_neighbours(text, search_tokenizer, search_model, token_dropout)
            end_time = time.time()
            total_generate_neighbours_time += (end_time - start_time)
            neighbors_data[text] = neighbors
    save_neighbors_to_csv(neighbors_data)
    print(f"Total generate_neighbours time: {total_generate_neighbours_time:.4f} seconds")

# Attack using loaded/generated neighbors
total_neighbor_logprobs_time = 0
for text, label in tqdm(zip(texts, labels), desc="Running Attack"): # tqdm for running attack
    orig_logprob = get_logprob(text, attack_tokenizer, attack_model)
    with torch.no_grad():
        start_time = time.time()
        neighbors = neighbors_data[text]
        neighbor_logprobs = [get_logprob(n, attack_tokenizer, attack_model) for n in neighbors]
        end_time = time.time()
        total_neighbor_logprobs_time += (end_time - start_time)

        avg_neighbor_logprob = np.mean(neighbor_logprobs)
        attack_score = orig_logprob - avg_neighbor_logprob
        all_attack_scores.append((attack_score, label))

print(f"Total neighbor_logprobs time: {total_neighbor_logprobs_time:.4f} seconds")

# ... (rest of the code: metrics, saving results, plotting)
true_labels = [label for _, label in all_attack_scores]
attack_scores = [score for score, _ in all_attack_scores]

# Remove NaN values
attack_scores = np.array(attack_scores)
nan_indices = np.isnan(attack_scores)

if np.any(nan_indices):
    print(f"Found NaN values in pred_scores at indices: {np.where(nan_indices)[0]}")
    attack_scores = attack_scores[~nan_indices]
    true_labels = np.array(true_labels)[~nan_indices]

if len(attack_scores) == 0:
    print("No valid predictions to calculate AUROC.")
else:

    # Calculate metrics
    auroc, fpr95, tpr05 = get_metrics(attack_scores, true_labels)
    print(f"AUROC: {auroc:.2%}")
    print(f"FPR at 95% TPR: {fpr95:.2%}")
    print(f"TPR at 5% FPR: {tpr05:.2%}")

    # Save results (same as before)
    results = {'method': ['logprob'], 'auroc': [f"{auroc:.1%}"], 'fpr95': [f"{fpr95:.1%}"], 'tpr05': [f"{tpr05:.1%}"]}
    df = pd.DataFrame(results)

    save_root = f"Neighbourhood/results_N50_xsum"
    os.makedirs(save_root, exist_ok=True)
    fname = f"{attack_model_name}_summary.csv"

    if os.path.isfile(os.path.join(save_root, fname)):
        df.to_csv(os.path.join(save_root, fname), index=False, mode='a', header=False)
    else:
        df.to_csv(os.path.join(save_root, fname), index=False)

    # Save full scores
    output_df = pd.DataFrame({'neighbor': attack_scores, 'labels': true_labels})
    output_df.to_csv(os.path.join(save_root, f"{attack_model_name}_full_scores.csv"), index=False)












