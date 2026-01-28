import numpy as np
import random
import torch

class DataIteratorLabel:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 padding_value=0,
                 train_flag=0,
                 train_neg_per_pos=0,
                 device=None
                ):
        """
        source: テキストファイル。各行に時系列順の item_id をスペース区切りで並べたもの
        batch_size: バッチ内のユーザー数
        maxlen: 履歴＋ターゲット長が maxlen+1 となるように切り出し
        padding_value: パディングに使う値
        train_flag: 0=学習モード（ランダム分割）、それ以外=評価モード（80/20分割）
        train_neg_per_pos: 正例ごとの負例サンプリング数
        device: torch.device
        """
        self.read(source)
        self.users = list(self.graph.keys())

        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.seq_len = maxlen + 1
        self.padding_value = padding_value
        self.train_neg_per_pos = train_neg_per_pos
        self.device = device or torch.device('cpu')
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # ユーザーサンプリング
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        sequence_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]
            # 正例: history + target
            if self.train_flag == 0:
                # ランダムに切り出しポイント k を選ぶ（少なくとも 4 番目以降）
                k = random.choice(range(4, len(item_list)))
                target = item_list[k]
                history = item_list[:k]
            else:
                # 評価時は最初の80%を history、残りを target
                k = int(len(item_list) * 0.8)
                target = item_list[k:]
                history = item_list[:k]

            # シーケンス結合
            seq = history + (target if isinstance(target, list) else [target])

            # 左パディング
            if len(seq) >= self.seq_len:
                seq = seq[-self.seq_len:]
            else:
                pad_len = self.seq_len - len(seq)
                seq = [self.padding_value] * pad_len + seq

            sequence_list.append(seq)

        # Positive Tensor
        pos_tensor = torch.tensor(sequence_list, dtype=torch.long, device=self.device)

        # Negative sampling
        if self.train_neg_per_pos > 0:
            if self.padding_value <= 1:
                raise ValueError("padding_value must be >1 for negative sampling")
            neg_shape = (self.batch_size, self.seq_len, self.train_neg_per_pos)
            neg_tensor = torch.randint(
                low=1,
                high=self.padding_value,
                size=neg_shape,
                dtype=torch.long,
                device=self.device
            )
        else:
            neg_tensor = None

        return pos_tensor, neg_tensor

    def read(self, source):
        """
        source: 各行がスペース区切りの item_id を持つテキストファイル
        """
        self.graph = {}
        self.items = set()

        with open(source, 'r') as f:
            for uid, line in enumerate(f):
                # 空行はスキップ
                tokens = line.strip().split()
                if not tokens:
                    continue
                seq = [int(tok) for tok in tokens]
                self.graph[uid] = seq
                self.items.update(seq)

        # user list, item list を作成
        # self.users は __init__ で graph.keys() を参照
        self.items = list(self.items)
