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

sys.path.append("../interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

sys.path.append("../predict_interest")
from eval_utils import evaluate
from utils import build_model_vec, get_device, load_config
from data_iterator_direct import DataIteratorDirect

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

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():
    print("test")
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--filter_rated',type=lambda x: x.lower() in ("true", "1", "yes"),default=True)

    parser.add_argument('--result_csv_path', default="./result.csv")
    parser.add_argument('--exp_name', default="direct_topk")

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

    config_gsasrec = load_config(args.config)
    device = get_device()
    check_point = dataset_stats['interest_gsasrec_model_path']

    model_gsasrec = build_model_vec(config_gsasrec)
    model_gsasrec = model_gsasrec.to(device)
    model_gsasrec.load_state_dict(torch.load(check_point, map_location=device))

    test_dataloader = DataIteratorDirect(
        source=config_gsasrec.test_full_path, 
        embedding_dir=config_gsasrec.embedding_dir_train,
        batch_size=config_gsasrec.eval_batch_size, 
        maxlen=config_gsasrec.sequence_length, 
        num_item=config_gsasrec.num_items, 
        device=device,
        train_flag=0,
    )

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

    with torch.no_grad():
        max_batches = len(test_dataloader)

        for batch_idx, (pos_emb, target, _, mask) in tqdm.tqdm(
            enumerate(test_dataloader), total=max_batches
        ):
            pos_emb = pos_emb.to(device)        # (B, T, D)
            target = target.to(device)          # (B,)
            mask = mask.to(device)              # (B, T)

            # ---- model forward ----
            last_hidden, _ = model_gsasrec(pos_emb[:, :-1, :], mask[:, :-1])   # (B, T-1, D)
            user_vec = last_hidden[:, -1, :]                           # (B, D)

            # ---- 全ユーザ共通 item_emb を用いてスコア計算 ----
            scores = torch.matmul(user_vec, item_embeddings.T)  # (B, num_items)

            # ---- topk ----
            values, indices = torch.topk(scores, 50, dim=1)  # (B, topk)

            # ---- IR measures 用データ構築 ----
            for items_pred, scores_pred, tgt in zip(indices, values, target):
                for item, score in zip(items_pred, scores_pred):
                    scored_docs.append(
                        ScoredDoc(str(users_processed), str(item.item()), score.item())
                    )
                qrels.append(Qrel(str(users_processed), str(tgt.item()), 1))
                users_processed += 1

    metrics = [nDCG@50, nDCG@20, nDCG@10, nDCG@5, nDCG@3, nDCG@1,
                R@50, R@20, R@10, R@5, R@3, R@1]
    result_muse = ir_measures.calc_aggregate(metrics, qrels, scored_docs)

    print(result_muse)
    save_result_to_csv(args.result_csv_path, args.exp_name, result_muse)
            
if __name__ == "__main__":
    main()