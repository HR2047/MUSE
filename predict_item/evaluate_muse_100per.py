import torch
import tqdm
import sys
from ir_measures import ScoredDoc, Qrel
from ir_measures import nDCG, R
import ir_measures
import time
import os
import json
from argparse import ArgumentParser

from dataset_utils import TestDataset
from torch.utils.data import Dataset, DataLoader

sys.path.append("/home/hirosawa/research_m/MUSE_ver4/interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

sys.path.append("/home/hirosawa/research_m/MUSE_ver4/predict_interest_pre")
from eval_utils import evaluate
from utils import build_model, get_device, load_config

import torch
import tensorflow as tf
from collections import Counter

def load_all_user_embeddings(user_embedding_dir, total_users, device):
    """全ての user_embedding を一度にロード（ディスクI/O削減）"""
    total_start = time.time()
    user_embeddings_dict = {}
    for idx in range(total_users):
        start = time.time()
        file_path = os.path.join(user_embedding_dir, f"interest_vec_{idx}.pt")
        if os.path.exists(file_path):  # ファイルが存在しない場合のチェック
            user_embeddings_dict[idx] = torch.load(file_path, map_location=device)
            end = time.time()
            if(idx%10000==0):
                print(f"Loaded user embedding: {idx} / {total_users}, time: {end - start:.4f} sec, total time: {end - total_start:.4f}")  # ログ出力
        else:
            print(f"警告: {file_path} が見つかりません")
            user_embeddings_dict[idx] = None
    return user_embeddings_dict

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():
    print("test")
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str)
    parser.add_argument('--config', type=str)

    args = parser.parse_args()

    dataset_stats = load_json(args.dataset_stats_path)

    # 訓練時に用いたパラメータ
    n_mid = dataset_stats['num_items']  # アイテムの数（IDの数）
    embedding_dim = dataset_stats['embedding_dim']  # 埋め込み次元
    hidden_size = dataset_stats['hidden_size']  # 隠層のサイズ
    batch_size = dataset_stats['batch_size']  # バッチサイズ
    num_interest = dataset_stats['num_interests']  # ユーザが持つ興味の数（埋め込み数）
    seq_len = dataset_stats['seq_len']  # 履歴の長さ

    model_comirec_path = dataset_stats['model_comirec_path']
    user_embedding_dir = dataset_stats['user_embedding_dir']
    num_user = dataset_stats['num_users']

    item_output_path = dataset_stats['data_test_output_path']
    interest_output_path = dataset_stats['interest_test_output_path']

    config_gsasrec = load_config(args.config)
    device = get_device()

    user_embedding_dict = load_all_user_embeddings(user_embedding_dir, num_user, device)
    gpu_options = tf.GPUOptions(allow_growth=True)

    config_comirec = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    # TensorFlow セッションの開始
    with tf.Session(config=config_comirec) as sess:
        model_comirec = Model_ComiRec_SA(n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len)
        
        # セッションの初期化
        sess.run(tf.global_variables_initializer())

        model_comirec.restore(sess, model_comirec_path)
        item_embeddings = model_comirec.output_item(sess)
        item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)

    item_embeddings = item_embeddings.to(device)

    eval_batch_size = config_gsasrec.eval_batch_size
    dataset = TestDataset(item_output_path, interest_output_path)
    test_dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    max_batches = len(test_dataloader)

    users_processed = 0
    scored_docs = []
    qrels = []
    
    start = time.time()

    # data = 正解, interest = 正解の興味
    for batch_idx, (data ,interest) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, interest = data.to(device), interest.to(device)
        
        data = data.long()
        ids = (interest - 1).long()

        batch_start = batch_idx * eval_batch_size

        # 各ユーザごとに user_embedding_dict から Tensor[k, D] を取り出してリストに
        batch_user_embeddings = [
            user_embedding_dict[batch_start + i][ ids[i] ]
            for i in range(ids.size(0))
        ]
        
        items = torch.zeros(interest.shape[0], 50)
        scores = torch.zeros(interest.shape[0], 50, dtype=torch.float32)
        
        for i in range(len(data)):
            # 内積計算
            inner_products = torch.matmul(batch_user_embeddings[i], item_embeddings.T)

            top_values, top_indices = torch.topk(inner_products, 50)
            items[i] = top_indices
            scores[i] = top_values
            
        for recommended_items, recommended_scores, data in zip(items, scores, data):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(data.item()), 1))
            users_processed += 1
            pass
        
    result_muse = ir_measures.calc_aggregate([nDCG@50, nDCG@20, nDCG@10, nDCG@5, nDCG@3, nDCG@1, R@50, R@20, R@10, R@5, R@3, R@1], qrels, scored_docs)
    evaluate_time = time.time() - start

    print(result_muse)
    print(f"time: ", evaluate_time)
            
if __name__ == "__main__":
    main()