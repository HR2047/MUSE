import torch
import tqdm
import sys
from ir_measures import ScoredDoc, Qrel, nDCG, R
import ir_measures
import time
import os
import json
from argparse import ArgumentParser

from dataset_utils import get_dataloader
from eval_utils import evaluate
from utils import build_model_label, get_device, load_config

sys.path.append("../interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

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
            # logging.info(f"Loaded user embedding: {file_path}, time: {end - start:.4f} sec, total time: {end - total_start:.4f}")  # ログ出力
        else:
            print(f"警告: {file_path} が見つかりません")
            user_embeddings_dict[idx] = None
    return user_embeddings_dict

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str)
    parser.add_argument('--config', type=str)

    parser.add_argument('--result_csv_path', default="log/result.csv")
    parser.add_argument('--exp_name', default="comirec")    

    args = parser.parse_args()

    dataset_stats = load_json(args.dataset_stats_path)

    num_user = dataset_stats['num_users']
    n_mid = dataset_stats['num_items']
    num_interest = dataset_stats['num_interests']
    batch_size = dataset_stats['batch_size']
    eval_batch_size = dataset_stats['eval_batch_size']
    seq_len = dataset_stats['seq_len']
    user_embedding_dir = dataset_stats['user_embedding_dir']
    embedding_dim = dataset_stats['embedding_dim']
    hidden_size = dataset_stats['hidden_size']

    test_input_path = dataset_stats['interest_test_input_path']
    test_output_path = dataset_stats['interest_test_output_path']

    model_comirec_path = dataset_stats['model_comirec_path']

    config_gsasrec = load_config(args.config)
    device = get_device()
    check_ppoint = dataset_stats['interest_gsasrec_model_path']

    test_dataloader = get_dataloader(input_path=test_input_path, output_path=test_output_path, batch_size=config_gsasrec.eval_batch_size, max_length=20, padding_value=0)

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

    # model_gsasrec.eval()
    users_processed = 0
    scored_docs = []
    qrels = [] 
    eval_batch_size = config_gsasrec.eval_batch_size

    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, target = data.to(device), target.to(device)
        
        # if batch_idx>0:
        #     break
        
        batch_start = batch_idx * eval_batch_size
        
        batch_user_embeddings = [
            user_embedding_dict[batch_start+i]
            for i in range(len(data))
        ]

        items = torch.zeros(eval_batch_size, len(batch_user_embeddings[0]))
        scores = torch.zeros(eval_batch_size, len(batch_user_embeddings[0]), dtype=torch.float32)
        
        for i in range(len(data)):
            top_values = []   # 最大の内積値を保存

            for idx, user_embedding in enumerate(batch_user_embeddings[i]):
                # 内積計算
                inner_products = torch.matmul(user_embedding, item_embeddings.T)

                # 最大値と次点の値・インデックスを取得
                temp_max_val, temp_max_idx = torch.max(inner_products, dim=0)

                top_values.append(temp_max_val)    # 直接 tensor のまま保存
            
            # 上位2個のインデックスと内積値を tensor に変換
            items[i] = torch.arange(1, len(batch_user_embeddings[0])+1)  
            scores[i] = torch.stack(top_values)
            
        for recommended_items, recommended_scores, target in zip(items, scores, target):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(target.item()), 1))
            users_processed += 1
            pass
        
    result_comirec = ir_measures.calc_aggregate([R@1, R@2, R@3, R@4], qrels, scored_docs)

    print(result_comirec)
    save_result_to_csv(args.result_csv_path, args.exp_name, result_comirec)

if __name__ == "__main__":
    main()