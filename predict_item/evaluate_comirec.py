import torch
import tqdm
import sys
from ir_measures import ScoredDoc, Qrel, nDCG, R
import ir_measures
import time
import os
from argparse import ArgumentParser

from dataset_utils import get_val_or_test_dataloader

sys.path.append("/home/hirosawa/research_m/MUSE_ver2/interest_extraction")
from model import Model_ComiRec_SA  # モデルが格納されているファイルをインポート

sys.path.append("/home/hirosawa/research_m/MUSE_ver2/predict_interest_pre")
from eval_utils import evaluate
from utils import build_model, get_device, load_config

import torch
import tensorflow as tf
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_all_user_embeddings(user_embedding_dir, total_users, device):
    """全ての user_embedding を一度にロード（ディスクI/O削減）"""
    user_embeddings_dict = {}
    for idx in range(total_users):
        file_path = os.path.join(user_embedding_dir, f"interest_vec_{idx}.pt")
        if os.path.exists(file_path):  # ファイルが存在しない場合のチェック
            user_embeddings_dict[idx] = torch.load(file_path, map_location=device)
            end = time.time()
            # logging.info(f"Loaded user embedding: {file_path}, time: {end - start:.4f} sec, total time: {end - total_start:.4f}")  # ログ出力
        else:
            print(f"警告: {file_path} が見つかりません")
            user_embeddings_dict[idx] = None
    return user_embeddings_dict

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str)

    args = parser.parse_args()

    dataset_stats = load_json(args.dataset_stats_path)

    num_users = dataset_stats['num_users']
    num_items = dataset_stats['num_items']
    num_interests = dataset_stats['num_interests']
    batch_size = dataset_stats['batch_size']
    eval_batch_size = dataset_stats['eval_batch_size']
    seq_len = dataset_stats['seq_len']
    user_embedding_dir = dataset_stats['user_embedding_dir']
    embedding_dim = dataset_stats['embedding_dim']
    hidden_size = dataset_stats['hidden_size']

    test_input_interest_path = dataset_stats['interest_test_input_path']
    test_output_path = dataset_stats['data_test_output_path']

    model_comirec_path = dataset_stats['model_comirec_path']

    device = get_device()

    # user_embedding取得
    user_embedding_dict = load_all_user_embeddings(user_embedding_dir, num_users, device)
    gpu_options = tf.GPUOptions(allow_growth=True)

    config_comirec = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    # item_embedding取得
    # TensorFlow セッションの開始
    with tf.Session(config=config_comirec) as sess:
        model_comirec = Model_ComiRec_SA(num_items, embedding_dim, hidden_size, batch_size, num_interests, seq_len)
        
        # セッションの初期化
        sess.run(tf.global_variables_initializer())

        model_comirec.restore(sess, model_comirec_path)
        item_embeddings = model_comirec.output_item(sess)
        item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)

    item_embeddings = item_embeddings.to(device)
    test_dataloader = get_val_or_test_dataloader(num_interests+1, input_file_path=test_input_interest_path, output_file_path=test_output_path, batch_size=eval_batch_size, max_length=seq_len)

    max_batches = len(test_dataloader)

    # model_gsasrec.eval()
    users_processed = 0
    scored_docs = []
    qrels = [] 
    
    start = time.time()

    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_dataloader), total=max_batches):

        data, target = data.to(device), target.to(device)
        
        # if batch_idx>0:
        #     break
        
        batch_start = batch_idx * eval_batch_size
        
        batch_user_embeddings = [
            user_embedding_dict[batch_start+i]
            for i in range(len(data))
        ]
        
        items = torch.zeros(eval_batch_size, 50)
        scores = torch.zeros(eval_batch_size, 50, dtype=torch.float32)
        
        for i in range(len(data)):
            all_inner_products = []
            
            for user_embedding in batch_user_embeddings[i]:
                # 内積計算
                inner_products = torch.matmul(user_embedding, item_embeddings.T)
                all_inner_products.append(inner_products)
                
            # [n, num_items] → 転置して [num_items, n] にして、各アイテムの最大値を取る
            all_inner_products = torch.stack(all_inner_products, dim=0)  # [n, num_items]
            max_inner_products, _ = torch.max(all_inner_products, dim=0)  # [num_items]

            # トップ50を取得（重複なし）
            top_values, top_indices = torch.topk(max_inner_products, 50)

            items[i] = top_indices
            scores[i] = top_values
            
        for recommended_items, recommended_scores, target in zip(items, scores, target):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(ScoredDoc(str(users_processed), str(int(item.item())), score.item()))
            qrels.append(Qrel(str(users_processed), str(target.item()), 1))
            users_processed += 1
            pass
        
    result = ir_measures.calc_aggregate([nDCG@50, nDCG@20, R@50, R@20, R@1], qrels, scored_docs)
    evaluate_time = time.time() - start

    print(result)
    print(f"time: ", evaluate_time)
            

if __name__ == "__main__":
    main()
        
    