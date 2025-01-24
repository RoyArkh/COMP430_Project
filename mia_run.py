import os, argparse

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from peft import PeftModel, PeftConfig


# Setting available device to GPU (if CUDA is available)
device = torch.device("cuda")
print(f"Device: {device}")

# Helper function to convert a HuggingFace dataset into a list of dictionaries
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]  # Extracting each sample from the dataset
        all_data.append(ex)  # Appending each sample to the list
    return all_data

# Argument parsing
parser = argparse.ArgumentParser()
# Root dataset and model are specified through arguments
parser.add_argument('--root_dataset', type=str, default="wiki")


# Model argument supports a variety of pre-trained models (including LoRA and standard fine-tuned models)
parser.add_argument('--model', type=str, default='mia-llm/MIA-GPT2-WikiText-Fine-Tuned-V2',
    choices=[
    "mia-llm/pythia-70m-wikitext2raw-roya",
    "mia-llm/pythia-160m-wikitext2raw-roya", "mia-llm/pythia-410m-wikitext2raw-roya", "mia-llm/pythia-1b-wikitext2raw-roya", "mia-llm/pythia-1.4b-wikitext2raw-roya",
    "mia-llm/gpt-neo-125m-wikitext2raw-roya", "mia-llm/gpt-neo-1.3B-wikitext2raw-roya",
    "mia-llm/MobileLLM-125M-wikitext2raw-hosein", "mia-llm/MobileLLM-350M-wikitext2raw-roya", "mia-llm/MobileLLM-600M-wikitext2raw-roya", "mia-llm/MobileLLM-1B-wikitext2raw-roya", "mia-llm/MobileLLM-1.5B-wikitext2raw-roya",
    "mia-llm/pythia-70m-xsum-roya", "mia-llm/pythia-160m-xsum-roya", "mia-llm/pythia-410m-xsum-roya", "mia-llm/pythia-1b-xsum-roya",
    "mia-llm/pythia-160m-AGnews-roya", "mia-llm/pythia-70m-AGnews-roya",
    "mia-llm/gpt-neo-125m-xsum-roya", "mia-llm/gpt-neo-1.3B-xsum-roya"
    ]
)

# Dataset options including WikiMIA benchmarks and perturbed/paraphrased versions
parser.add_argument(
    '--dataset', type=str, default='dataset_32', 
    choices=[
        'WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128',
        'dataset_32','dataset_64','dataset_128',
        'paraphrased_dataset_32','paraphrased_dataset_64','paraphrased_dataset_128',
        'perturbed_dataset_32','perturbed_dataset_64','perturbed_dataset_128',
        'perturbed_paraphrased_dataset_32','perturbed_paraphrased_dataset_64','perturbed_paraphrased_dataset_128',
        'concat_dataset_32', "h0ssn/Agnews_MIA_benchmark_RH", "h0ssn/Xsum_MIA_benchmark_RH"
    ]
)

# Additional arguments for experiment configuration
parser.add_argument("--base_model", default=False)
parser.add_argument("--experiment_id", default=None, type=str)
parser.add_argument("--is_main", default=False)
parser.add_argument("--lora", default=False)  # Indicates whether to use a LoRA fine-tuned model
parser.add_argument('--half', action='store_true')  # Enables half-precision model loading
parser.add_argument('--int8', action='store_true')  # Enables 8-bit model loading
args = parser.parse_args()


# Function to load the model (and tokenizer) based on arguments
def load_model(name, ref=False):
    int8_kwargs = {}
    half_kwargs = {}
    
    # If int8 or half-precision is selected, pass these arguments (except for ref models)
    if args.int8 and not ref:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half and not ref:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
        
    # Check for MobileLLM-specific handling
    if "MobileLLM" in name:
        print("Loading MobileLLM model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True, **int8_kwargs, **half_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True, use_fast=False
            )
            tokenizer.add_special_tokens({
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            })
        except Exception as e:
            raise RuntimeError(f"Failed to load MobileLLM model or tokenizer: {e}")
    else:
        # Standard model loading for other models
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # Ensure model is in evaluation mode
    model.eval()
    
    return model, tokenizer


# Load the selected model and tokenizer
model, tokenizer = load_model(args.model)
# Load the reference model and tokenizer for reference attack
if "mobilellm" in args.model.lower():
    ref_model, ref_tokenizer = load_model('facebook/MobileLLM-1.5B', ref=True)
elif "pythia" in args.model.lower():
    ref_model, ref_tokenizer = load_model('EleutherAI/pythia-2.8b', ref=True)
elif "gpt-neo" in args.model.lower():
    ref_model, ref_tokenizer = load_model('EleutherAI/gpt-neo-2.7B', ref=True)
else:
    print("Couldn't load reference Model")

#ref_model, ref_tokenizer = load_model('openai-community/gpt2-large', ref=True)

# Dataset loading based on the selected dataset and root dataset
using_concat_dataset = False
wikimia_active = False

# Loading the appropriate dataset based on the root dataset and specific choice

if 'WikiMIA' in args.dataset:
    dataset = load_dataset('swj0419/WikiMIA', split=args.dataset)
    wikimia_active = True
elif 'concat' in args.dataset:
    dataset = load_dataset('mia-llm/wikitext2-MIA-Benchmark-Concat', split=args.dataset)  
    using_concat_dataset = True
# Additional dataset loading options for AG News and XSum
elif args.root_dataset == "ag_news":
    dataset = load_dataset('mia-llm/AGnews-MIA-Benchmark',split=args.dataset)
elif args.root_dataset == "x_sum":
    dataset = load_dataset('mia-llm/xsum-MIA-Benchmark',split=args.dataset)
else:
    if args.is_main:
        dataset = load_dataset('mia-llm/wikitext2-MIA-Benchmark',split=args.dataset)
    else:
        dataset = load_dataset('mia-llm/wikitext2-MIA-Benchmark', revision="V1",split=args.dataset)


# Convert the HuggingFace dataset to a list of dictionaries
data = convert_huggingface_data_to_list_dic(dataset)


# If not using the concat attack dataset, load perturbed/paraphrased datasets for MIA attacks
if not using_concat_dataset:
    if wikimia_active:
        perturbated_dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset + '_perturbed')
    else:
        if args.root_dataset == "ag_news":
            dataset = load_dataset('mia-llm/AGnews-MIA-Benchmark',split="paraphrased_"+args.dataset)
            perturbated_dataset = load_dataset('mia-llm/AGnews-MIA-Benchmark',split="perturbed_"+args.dataset)
        elif args.root_dataset == "x_sum":
            dataset = load_dataset('mia-llm/xsum-MIA-Benchmark',split="paraphrased_"+args.dataset)
        else:
            perturbated_dataset = load_dataset('mia-llm/wikitext2-MIA-Benchmark', revision="V1",split="perturbed_paraphrased_"+args.dataset)
    # print(f"Dataset being loaded: {args.root_dataset}")
    # print(f"Perturbated Dataset Split: {perturbated_dataset}")
    '''
    # Convert perturbed dataset to list
    perturbed_data = convert_huggingface_data_to_list_dic(perturbated_dataset)
    #num_neighbors = len(perturbed_data) // len(data)
    num_neighbors = 4
    '''
else:
    labels = [d['label'] for d in data]
    print(labels)


# Inference function: computes log-likelihood for a given text
def inference(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Encode text into token ids
    input_ids = input_ids.to(device)  # Move to GPU
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(input_ids, labels=input_ids)  # Perform forward pass
    loss, logits = outputs[:2]  # Extract loss and logits
    ll = -loss.item()  # Log-likelihood is the negative of the loss
    return ll


# Scoring different aspects for each input sample
scores = defaultdict(list)
if using_concat_dataset:
    chunk_labels = []
    for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
        full_text = d['text']
        # Validate that text length is consistent with chunk labels
        assert len(full_text.split(' ')) // 32 == len(labels[i]), \
            f"{i}, {len(full_text.split(' '))} != {len(labels[i])}"
        tmp = full_text.split(' ')
        text_chunks = [' '.join(tmp[j*32:(j+1)*32]) for j in range(len(labels[i]))]

        # Perform inference on the entire text
        input_ids = torch.tensor(tokenizer.encode(full_text)).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        # Validate the token lengths for each chunk
        chunk_token_lens = []
        for j, chunk in enumerate(text_chunks):
            tmp_ids = tokenizer.encode(chunk)
            if j > 0:
                chunk_token_lens.append(len(tmp_ids) - 1)
            else:
                chunk_token_lens.append(len(tmp_ids))
        assert sum(chunk_token_lens) == len(input_ids[0]), \
            f"{i}, {sum(chunk_token_lens)} != {len(input_ids[0])}"

        # Calculate log-likelihood and Mink/Mink++ scores for each chunk
        for j, chunk in enumerate(text_chunks):
            chunk_input_ids = input_ids[0][sum(chunk_token_lens[:j]):sum(chunk_token_lens[:j+1])]
            chunk_logits = logits[0][sum(chunk_token_lens[:j]):sum(chunk_token_lens[:j+1])]

            chunk_input_ids = chunk_input_ids[1:].unsqueeze(-1)
            chunk_logits = chunk_logits[:-1]

            loss = F.cross_entropy(
                chunk_logits.unsqueeze(0).permute(0, 2, 1), 
                chunk_input_ids[:, 0].unsqueeze(0)
            )
            ll = -loss.item() # log-likelihood

            # assuming the score is larger for training data
            # and smaller for non-training data
            # this is why sometimes there is a negative sign in front of the score
            scores['loss'].append(ll)
            scores['zlib'].append(
                ll / len(zlib.compress(bytes(chunk, 'utf-8')))
            )

            #ll_ref = inference(text, ref_model)
            ll_lowercase = inference(text.lower(), model)
            #scores['ref'].append(ll - ll_ref)
            scores['lowercase'].append(ll_lowercase / ll)


            # mink and mink++
            probs = F.softmax(chunk_logits, dim=-1)
            log_probs = F.log_softmax(chunk_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=chunk_input_ids).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            ## mink
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = int(len(token_log_probs) * ratio)
                topk = np.sort(token_log_probs.cpu())[:k_length]
                scores[f'mink_{ratio}'].append(np.mean(topk).item())
                
            ## mink++
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = int(len(mink_plus) * ratio)
                topk = np.sort(mink_plus.cpu())[:k_length]
                scores[f'mink++_{ratio}'].append(np.mean(topk).item())

            chunk_labels.append(labels[i][j])
else:
    for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')):
        if "dataset" in args.dataset:
            text = d['text']
        else:
            text = d['text']
        
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ll = -loss.item() # log-likelihood
        ll_ref = inference(text, ref_model)
        ll_lowercase = inference(text.lower(), model)

        '''
        ll_neighbors = []
        for j in range(num_neighbors):
            if "dataset" in args.dataset:
                text = perturbed_data[i * num_neighbors + j]['text']
            else:
                text = perturbed_data[i * num_neighbors + j]['input']
            ll_neighbors.append(inference(text, model))
        '''
        # assuming the score is larger for training data
        # and smaller for non-training data
        # this is why sometimes there is a negative sign in front of the score
        '''
        scores['neighbor'].append(ll - np.mean(ll_neighbors))
        '''
        scores['ref'].append(ll - ll_ref)
        scores['lowercase'].append(ll_lowercase / ll)

        
        
        # loss and zlib
        scores['loss'].append(ll)
        scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

        # mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        ## mink
        for ratio in [0.2]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())

        ## mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.8]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'mink++_{ratio}'].append(np.mean(topk).item())



# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, score in scores.items():
    if using_concat_dataset:
        auroc, fpr95, tpr05 = get_metrics(score, chunk_labels)
    else:
        auroc, fpr95, tpr05 = get_metrics(score, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

output_df = pd.DataFrame.from_dict(scores)

# display results
df = pd.DataFrame(results)
print(df)

save_root = f"results/latest_xsum" #roya note
if not os.path.exists(save_root):
    os.makedirs(save_root)

model_id = args.model.split('/')[-1]

fname = f"{model_id}.csv"

if args.base_model:
    fname = f"{model_id}-{args.experiment_id}.csv"

if os.path.isfile(os.path.join(save_root, fname)):
    df.to_csv(os.path.join(save_root,fname), index=False, mode='a', header=False)
else:
    df.to_csv(os.path.join(save_root, fname), index=False)

score_path = f"{model_id}_full_scores.csv"
if args.base_model:
    score_path = f"{model_id}-{args.experiment_id}_full_scores.csv"

if os.path.isfile(os.path.join(save_root, score_path)):
    output_df.to_csv(os.path.join(save_root, score_path), index=False, mode='a', header=False)
else:
    output_df.to_csv(os.path.join(save_root, score_path), index=False)