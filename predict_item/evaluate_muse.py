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

from dataset_utils_muse import get_val_or_test_dataloader

sys.path.append("/home/hirosawa/research_m/MUSE_ver2/interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

sys.path.append("/home/hirosawa/research_m/MUSE_ver2/predict_interest_pre")
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
    parser.add_argument('--filter_rated', default=True)

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

    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, target = data.to(device), target.to(device)
        
        # if batch_idx>0:
        #     break

        interests, scores = model_gsasrec.get_predictions(data, 1)
        
        batch_start = batch_idx * eval_batch_size
        
        batch_user_embeddings = [
            user_embedding_dict[batch_start+i][interests[i].item()-1]
            for i in range(len(data))
        ]
        
        items = torch.zeros(interests.shape[0], 50)
        scores = torch.zeros(interests.shape[0], 50, dtype=torch.float32)
        
        for i in range(len(data)):
            # 内積計算
            inner_products = torch.matmul(batch_user_embeddings[i], item_embeddings.T)

            # rated に含まれるアイテムを除外
            if(args.filter_rated):
                rated_items = rated[i]
                for rated_item in rated_items:
                    if rated_item < inner_products.size(0):
                        inner_products[rated_item] = float('-inf')
                        
            top_values, top_indices = torch.topk(inner_products, 50)
            items[i] = top_indices
            scores[i] = top_values
            
        for recommended_items, recommended_scores, target in zip(items, scores, target):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(target.item()), 1))
            users_processed += 1
            pass
        
    result_muse = ir_measures.calc_aggregate([nDCG@50, nDCG@20, R@50, R@20, R@1], qrels, scored_docs)

    print(result_muse)
            
if __name__ == "__main__":
    main()