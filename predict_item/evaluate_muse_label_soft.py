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
import torch.nn.functional as F

from dataset_utils_muse import get_val_or_test_dataloader

sys.path.append("../interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

sys.path.append("../predict_interest")
from eval_utils import evaluate
from utils import build_model, get_device, load_config

import torch
import tensorflow as tf
from collections import Counter

import csv

def save_result_to_csv(csv_path, row_name, result_dict, extra_info=None):
    row = {}

    # 行名
    row["exp_name"] = row_name

    # 追加情報
    if extra_info is not None:
        row.update(extra_info)

    # metric（順序は一旦無視）
    for metric in result_dict:
        row[str(metric)] = float(result_dict[metric])

    # -----------------------------
    # 既存CSVがあるかどうか
    # -----------------------------
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # 既存の列名

        # 既存列順に合わせる
        ordered_row = {}

        for col in header:
            ordered_row[col] = row.get(col, "")

        # 新しい列（後から追加されたmetricなど）
        new_cols = [k for k in row.keys() if k not in header]
        for col in new_cols:
            ordered_row[col] = row[col]

        fieldnames = header + new_cols

    else:
        # CSVが存在しない場合（初回）
        ordered_row = row
        fieldnames = list(row.keys())

    # -----------------------------
    # 書き込み
    # -----------------------------
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(ordered_row)

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
    parser.add_argument('--filter_rated', default=True)

    parser.add_argument('--result_csv_path', default="./result.csv")
    parser.add_argument('--exp_name', default="label_soft")

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

    test_input_path = dataset_stats['interest_test_input_path']
    test_output_path = dataset_stats['data_test_output_path']

    rated_path = dataset_stats['data_test_input_path']

    config_gsasrec = load_config(args.config)
    device = get_device()
    check_point = dataset_stats['interest_gsasrec_model_path']

    model_gsasrec = build_model(config_gsasrec)
    model_gsasrec = model_gsasrec.to(device)
    model_gsasrec.load_state_dict(torch.load(check_point, map_location=device))

    test_dataloader = get_val_or_test_dataloader(num_interest+1, input_file_path=test_input_path, output_file_path=test_output_path, rated_file_path = rated_path, batch_size=config_gsasrec.eval_batch_size, max_length=20)

    user_embedding_dict = load_all_user_embeddings(user_embedding_dir, num_user, device)
    gpu_options = tf.GPUOptions(allow_growth=True)

    config_comirec = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    # TensorFlow セッションの開始
    with tf.Session(config=config_comirec) as sess:
        model_comirec = Model_ComiRec_SA(n_mid+1, embedding_dim, hidden_size, batch_size, num_interest, seq_len)
        
        # セッションの初期化
        sess.run(tf.global_variables_initializer())

        model_comirec.restore(sess, model_comirec_path)
        item_embeddings = model_comirec.output_item(sess)
        item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)

    item_embeddings = item_embeddings.to(device)

    max_batches = len(test_dataloader)

    model_gsasrec.eval()
    users_processed = 0
    scored_docs = []
    qrels = [] 
    eval_batch_size = config_gsasrec.eval_batch_size
    
    start = time.time()

    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, target = data.to(device), target.to(device)
        
        interests, interests_scores = model_gsasrec.get_predictions(data, num_interest)
        
        _, indices = interests.sort(dim=1)
        interests_scores_sorted = torch.gather(interests_scores, dim=1, index=indices)
        interests_scores_norm = F.softmax(interests_scores_sorted, dim=1)

        batch_start = batch_idx * eval_batch_size

        batch_user_embeddings = [
            user_embedding_dict[batch_start+i]
            for i in range(len(data))
        ]
        
        items = torch.zeros(interests.shape[0], 50)
        scores = torch.zeros(interests.shape[0], 50, dtype=torch.float32)
        
        for i in range(len(data)):
            all_inner_products = []
            
            for user_embedding in batch_user_embeddings[i]:
                # 内積計算
                inner_products = torch.matmul(user_embedding, item_embeddings.T)
                all_inner_products.append(inner_products)
                
            # [n, num_items] → 転置して [num_items, n] にして、各アイテムの最大値を取る
            all_inner_products = torch.stack(all_inner_products, dim=0)  # [n, num_items]
            # max_inner_products, _ = torch.max(all_inner_products, dim=0)  # [num_items]
            
            # ここで各行 k に interests_scores_norm[i][k] をかける
            w = interests_scores_norm[i].unsqueeze(1)       # → [num_interests, 1]
            weighted_inner_products = all_inner_products * w  # → [num_interests, num_items]
            sum_inner_products = weighted_inner_products.sum(dim=0)  # → [num_items]

            # rated に含まれるアイテムを除外
            if(args.filter_rated):
                rated_items = rated[i]
                for rated_item in rated_items:
                    if rated_item < inner_products.size(0):
                        sum_inner_products[rated_item] = float('-inf')

            # トップ50を取得（重複なし）
            top_values, top_indices = torch.topk(sum_inner_products, 50)

            items[i] = top_indices
            scores[i] = top_values
            
        for recommended_items, recommended_scores, target in zip(items, scores, target):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(target.item()), 1))
            users_processed += 1
            pass
        
    result_muse = ir_measures.calc_aggregate([nDCG@50, nDCG@20, nDCG@10, nDCG@5, nDCG@3, nDCG@1, R@50, R@20, R@10, R@5, R@3, R@1], qrels, scored_docs)
    evaluate_time = time.time() - start

    print(result_muse)
    print(f"time: ", evaluate_time)

    save_result_to_csv(args.result_csv_path, args.exp_name, result_muse)
            
if __name__ == "__main__":
    main()