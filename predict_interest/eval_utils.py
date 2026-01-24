import torch
import tqdm 
from gsasrec_label import GSASRec
from gsasrec_vector import GSASRecVec
from ir_measures import ScoredDoc, Qrel
import ir_measures

# model(入力：label 出力：label), 出力：ir_measures
def evaluate(model: GSASRec, data_loader, metrics, limit, filter_rated, device):
    model.eval()
    users_processed = 0
    scored_docs = []
    qrels = [] 
    with torch.no_grad():
        max_batches = len(data_loader)
        for batch_idx, (data, rated, target) in tqdm.tqdm(enumerate(data_loader), total=max_batches):
            data, target = data.to(device), target.to(device)
            if filter_rated:
                items, scores = model.get_predictions(data, limit, rated)
            else:
                items, scores = model.get_predictions(data, limit)
            for recommended_items, recommended_scores, target in zip(items, scores, target):
                for item, score in zip(recommended_items, recommended_scores):
                    scored_docs.append(ScoredDoc(str(users_processed), str(item.item()), score.item()))
                qrels.append(Qrel(str(users_processed), str(target.item()), 1))
                users_processed += 1
                pass
    result = ir_measures.calc_aggregate(metrics, qrels, scored_docs)
    return result
    
# model(入力：embedding　出力：embedding), 出力：スコア平均
def evaluate_vec(model, data_loader, metrics, topk, device):
    """
    pos_emb: (B, T, D)
    target:  (B,)
    item_emb: list of each user's embedding table
    mask:    (B, T)
    """

    model.eval()
    users_processed = 0
    scored_docs = []
    qrels = []

    with torch.no_grad():
        max_batches = len(data_loader)

        for batch_idx, (pos_emb, target, item_emb_list, mask) in tqdm.tqdm(
            enumerate(data_loader), total=max_batches
        ):
            pos_emb = pos_emb.to(device)        # (B, T, D)
            target = target.to(device)          # (B,)
            mask = mask.to(device)              # (B, T)

            last_hidden, _ = model(pos_emb[:, :-1, :], mask[:, :-1])   # (B, T-1, D)
            user_vec = last_hidden[:, -1, :]                           # (B, D)

            batch_scores = []
            batch_items = []

            B = pos_emb.size(0)

            for b in range(B):
                emb_table = item_emb_list[b]      # (num_items+1, D)
                # padding 行(0番) を除外
                item_emb = emb_table[1:, :]       # (num_items, D)

                # ユーザベクトル → item score
                scores = torch.matmul(item_emb, user_vec[b])   # (num_items,)
                
                # 上位 topk を取得
                values, indices = torch.topk(scores, topk)
                
                # item_id は 0-index なのでそのまま扱う
                batch_scores.append(values)
                batch_items.append(indices+1)

            for items_pred, scores_pred, tgt in zip(batch_items, batch_scores, target):
                # 推薦されたアイテムのスコアを追加
                for item, score in zip(items_pred, scores_pred):
                    scored_docs.append(
                        ScoredDoc(str(users_processed), str(item.item()), score.item())
                    )
                # 正例
                qrels.append(Qrel(str(users_processed), str(tgt.item()), 1))
                users_processed += 1

    result = ir_measures.calc_aggregate(metrics, qrels, scored_docs)
    return result
