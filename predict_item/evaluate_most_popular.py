import torch
import tqdm
import sys
from ir_measures import ScoredDoc, Qrel, nDCG, R
import ir_measures
from argparse import ArgumentParser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from dataset_utils import get_val_or_test_dataloader

sys.path.append("/home/hirosawa/research_m/MUSE_ver2/predict_interest_pre")
from utils import get_device

import torch
import tensorflow as tf
from collections import Counter

import json

def get_top_50_numbers(file_path):
    all_numbers = []
    
    with open(file_path, 'r') as file:
        for line in file:
            numbers_in_line = line.strip().split()
            for num in numbers_in_line:
                try:
                    all_numbers.append(int(num))
                except ValueError:
                    pass  # 数字でない場合は無視

    counter = Counter(all_numbers)
    total_numbers = sum(counter.values())
    top_50 = counter.most_common(50)

    top_numbers = [num for num, count in top_50]
    top_frequencies = [count / total_numbers for _, count in top_50]

    return top_numbers, top_frequencies

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str)
    parser.add_argument('--test_input_path', type=str)
    parser.add_argument('--test_output_path', type=str)
    parser.add_argument('--metrics', type=list, default=[nDCG@50, nDCG@20, nDCG@10, nDCG@5, nDCG@3, nDCG@1, R@50, R@20, R@10, R@5, R@3, R@1])

    args = parser.parse_args()

    dataset_stats = load_json(args.dataset_stats_path)

    num_users = dataset_stats['num_users']
    num_items = dataset_stats['num_items']
    batch_size = dataset_stats['batch_size']
    eval_batch_size = dataset_stats['eval_batch_size']
    seq_len = dataset_stats['seq_len']

    device = get_device()
    
    top_numbers, top_frequencies = get_top_50_numbers(args.test_input_path)

    test_dataloader = get_val_or_test_dataloader(num_items+1, 
                                                 input_file_path=args.test_input_path, 
                                                 output_file_path=args.test_output_path, 
                                                 batch_size=eval_batch_size, 
                                                 max_length=seq_len)
    
    items = torch.tensor([top_numbers] * 50)
    scores = torch.tensor([top_frequencies] * 50)

    max_batches = len(test_dataloader)

    users_processed = 0
    scored_docs = []
    qrels = [] 

    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, target = data.to(device), target.to(device)
            
        for recommended_items, recommended_scores, target in zip(items, scores, target):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(target.item()), 1))
            users_processed += 1
            pass
        
    result = ir_measures.calc_aggregate(args.metrics, qrels, scored_docs)
    print(result)

if __name__ == "__main__":
    main()
        
    