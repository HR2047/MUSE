import torch
import tqdm
import sys
import time
import os
import json
from argparse import ArgumentParser

from ir_measures import ScoredDoc, Qrel, nDCG, R, calc_aggregate
from dataset_utils import get_val_or_test_dataloader

sys.path.append("/home/hirosawa/research_m/MUSE_ver3/interest_extraction")
from model import Model_ComiRec_SA

sys.path.append("/home/hirosawa/research_m/MUSE_ver3/predict_interest_pre")
from eval_utils import evaluate
from utils import build_model, get_device, load_config

import tensorflow as tf

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
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def beam_search_interests(model, data, beam_width, beam_length, seq_len):
    """
    Efficient batch-wise BeamSearch for interest prediction.
    - model: GSASRec モデルオブジェクト（バッチ入力対応 get_predictions）
    - data: Tensor (batch_size, seq_len)
    - beam_width: B
    - beam_length: L
    - seq_len: 元シーケンス長
    -> Tensor (batch_size,) of first predicted interest ID
    """
    batch_size = data.size(0)
    device = data.device

    # ビームのシーケンスリストとスコアリスト
    beams_seqs = [[data[i].unsqueeze(0)] for i in range(batch_size)]  # 各: list of tensors (1, seq_len)
    beams_scores = [[0.0] for _ in range(batch_size)]                # 各: list of floats

    for _ in range(beam_length):
        # 全ビームをまとめて予測
        all_seqs = torch.cat([seq for seqs in beams_seqs for seq in seqs], dim=0)  # (batch_size * B_curr, seq_len)
        all_interests, all_scores = model.get_predictions(all_seqs, beam_width)  # shapes: (N, B)

        new_beams_seqs = []
        new_beams_scores = []
        idx = 0
        for i in range(batch_size):
            current_seqs = beams_seqs[i]
            current_scores = beams_scores[i]
            candidates = []
            for b_idx, (seq, cum_score) in enumerate(zip(current_seqs, current_scores)):
                row_ints = all_interests[idx]     # shape (B,)
                row_scores = all_scores[idx]      # shape (B,)
                idx += 1
                for j in range(beam_width):
                    int_id = row_ints[j].item()
                    sc = row_scores[j].item()
                    # 左シフト＆append
                    next_seq = torch.roll(seq, shifts=-1, dims=1)
                    next_seq[0, -1] = int_id
                    candidates.append((next_seq, cum_score + sc))
            # 上位 B を保持
            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:beam_width]
            new_beams_seqs.append([seq for seq, _ in top])
            new_beams_scores.append([score for _, score in top])

        beams_seqs, beams_scores = new_beams_seqs, new_beams_scores

    # 最良系列の最初の予測 interest を抽出
    top_first = torch.zeros(batch_size, dtype=torch.long, device=device)
    first_idx = seq_len - beam_length
    for i in range(batch_size):
        best_seq = beams_seqs[i][0]
        top_first[i] = best_seq[0, first_idx]
    return top_first


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_stats_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--beam_width', type=int, required=True, help='Beam search width (B)')
    parser.add_argument('--beam_length', type=int, required=True, help='Beam search length (L)')
    args = parser.parse_args()

    # stats と設定読み込み
    stats = load_json(args.dataset_stats_path)

    # データセット設定
    n_mid = stats['num_items']
    embedding_dim = stats['embedding_dim']
    hidden_size = stats['hidden_size']
    batch_size = stats['batch_size']
    num_interest = stats['num_interests']
    seq_len = stats['seq_len']

    # パス設定
    user_embedding_dir = stats['user_embedding_dir']
    num_user = stats['num_users']
    model_comirec_path = stats['model_comirec_path']
    test_input_path = stats['interest_test_input_path']
    test_output_path = stats['data_test_output_path']
    gsasrec_ckpt = stats['interest_gsasrec_model_path']

    # GSASRec モデル構築
    config_gsasrec = load_config(args.config)
    device = get_device()
    model_gsasrec = build_model(config_gsasrec).to(device)
    model_gsasrec.load_state_dict(torch.load(gsasrec_ckpt, map_location=device))
    model_gsasrec.eval()

    # テストデータローダ
    test_loader = get_val_or_test_dataloader(
        num_interest+1,
        input_file_path=test_input_path,
        output_file_path=test_output_path,
        batch_size=config_gsasrec.eval_batch_size,
        max_length=seq_len
    )

    # ユーザ・アイテム埋め込み読み込み
    user_embeddings = load_all_user_embeddings(user_embedding_dir, num_user, device)
    gpu_opts = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opts)
    with tf.Session(config=tf_config) as sess:
        model_comirec = Model_ComiRec_SA(n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len)
        sess.run(tf.global_variables_initializer())
        model_comirec.restore(sess, model_comirec_path)
        item_emb_np = model_comirec.output_item(sess)
    item_embeddings = torch.tensor(item_emb_np, dtype=torch.float32, device=device)

    # 評価用バッファ
    scored_docs = []
    qrels = []
    user_counter = 0
    total_samples = 0
    changed_count = 0

    # バッチごとに評価
    for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        data = data.to(device)
        batch_size_actual = data.size(0)

        # 通常の1-step予測 (B=1) と BeamSearch 予測を取得
        # ベースライン
        base_interests, _ = model_gsasrec.get_predictions(data, 1)
        # BeamSearch
        beam_interests = beam_search_interests(model_gsasrec, data.clone(), args.beam_width, args.beam_width, seq_len)

        # 比較用カウンタ更新
        for i in range(batch_size_actual):
            total_samples += 1
            if base_interests[i].item() != beam_interests[i].item():
                changed_count += 1

        # BeamSearch の結果で推薦評価
        for i in range(batch_size_actual):
            global_user_idx = batch_idx * config_gsasrec.eval_batch_size + i
            emb_vecs = user_embeddings[global_user_idx]
            u_emb = emb_vecs[beam_interests[i].item()-1]
            inner = torch.matmul(u_emb, item_embeddings.T)
            top_vals, top_idxs = torch.topk(inner, 50)
            for item_id, score in zip(top_idxs, top_vals):
                scored_docs.append(ScoredDoc(str(user_counter), str(int(item_id.item())), score.item()))
            qrels.append(Qrel(str(user_counter), str(int(target[i].item())), 1))
            user_counter += 1

    # 指標計算
    results = calc_aggregate([nDCG@50, nDCG@20, R@50, R@20, R@1], qrels, scored_docs)
    print(results)

    # 変更数と割合出力
    print(f"BeamSearch と baseline の最初の interest 予測が変化したサンプル数: {changed_count}")
    print(f"総サンプル数: {total_samples}, 変化割合: {changed_count / total_samples:.2%}")

if __name__ == '__main__':
    main()
