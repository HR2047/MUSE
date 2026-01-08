import os
import torch
import random

class DataIteratorVer5:

    def __init__(
        self,
        source,
        embedding_dir,          # ← 1本化
        batch_size,
        maxlen,
        padding_value,
        train_neg_per_pos,
        device,
        train_flag=1
    ):
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.seq_len = maxlen + 1
        self.padding_value = padding_value
        self.train_neg_per_pos = train_neg_per_pos
        self.train_flag = train_flag   # 1=train, 0=eval

        self.embedding_dir = embedding_dir
        self.embedding_cache = {}

        self.graph = self._read_sequence(source)
        self.users = list(self.graph.keys())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.train_flag == 1:
            return self._next_train()
        else:
            return self._next_eval()

    # =====================================================
    # Train
    # =====================================================
    def _next_train(self):
        user_batch = random.sample(self.users, self.batch_size)
        seq_batch = []

        for user_id in user_batch:
            item_list = self.graph[user_id]
            k = random.choice(range(4, len(item_list)))
            history = item_list[:k]
            target = item_list[k]

            seq = history + [target]
            if len(seq) >= self.seq_len:
                seq = seq[-self.seq_len:]
            else:
                seq = [self.padding_value] * (self.seq_len - len(seq)) + seq
            seq_batch.append(seq)

        pos_ids = torch.tensor(seq_batch, dtype=torch.long, device=self.device) - 1

        # --- negative sampling ---
        if self.train_neg_per_pos > 0:
            neg_shape = (pos_ids.size(0), pos_ids.size(1), self.train_neg_per_pos)
            neg_ids = torch.randint(1, self.padding_value, neg_shape, device=self.device) - 1
        else:
            neg_ids = None

        pos_emb_batches, neg_emb_batches = [], []

        for b, user_id in enumerate(user_batch):
            emb_table = self._load_user_embedding(user_id)

            pos_emb_batches.append(emb_table[pos_ids[b]])
            if neg_ids is not None:
                neg_emb_batches.append(emb_table[neg_ids[b]])

        pos_emb = torch.stack(pos_emb_batches, 0)
        neg_emb = torch.stack(neg_emb_batches, 0) if neg_ids is not None else None

        mask = (pos_ids != self.padding_value - 1).float()
        return pos_emb, neg_emb, mask

    # =====================================================
    # Eval
    # =====================================================
    def _next_eval(self):
        if self.index >= len(self.users):
            self.index = 0
            raise StopIteration

        user_batch = self.users[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        seq_batch, target_batch, emb_tables = [], [], []

        for user_id in user_batch:
            item_list = self.graph[user_id]
            emb_table = self._load_user_embedding(user_id)
            emb_tables.append(emb_table)

            history = item_list[:-1]
            target = item_list[-1]

            history = [i - 1 for i in history]
            target = target - 1

            seq = history + [target]
            if len(seq) >= self.seq_len:
                seq = seq[-self.seq_len:]
            else:
                seq = [self.padding_value] * (self.seq_len - len(seq)) + seq

            seq_batch.append(seq)
            target_batch.append(target)

        seq_tensor = torch.tensor(seq_batch, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(target_batch, dtype=torch.long, device=self.device)

        mask = (seq_tensor != self.padding_value).float()

        pos_emb_batches = []
        for b in range(len(user_batch)):
            emb_table = emb_tables[b]
            idx = seq_tensor[b].clone()
            idx[idx == self.padding_value] = -1
            idx = idx + 1
            idx = idx.clamp(0, emb_table.size(0) - 1)
            pos_emb_batches.append(emb_table[idx])

        pos_emb = torch.stack(pos_emb_batches, 0)
        return pos_emb, target_tensor, emb_tables, mask

    # =====================================================
    # Utils
    # =====================================================
    def _read_sequence(self, source):
        graph = {}
        with open(source) as f:
            for uid, line in enumerate(f):
                tokens = line.strip().split()
                if tokens:
                    graph[uid] = [int(t) for t in tokens]
        return graph

    def _load_user_embedding(self, user_id):
        if user_id in self.embedding_cache:
            return self.embedding_cache[user_id]

        path = os.path.join(self.embedding_dir, f"interest_vec_{user_id}.pt")
        emb = torch.load(path).to(self.device)

        pad_emb = torch.zeros(1, emb.size(1), device=self.device)
        emb_table = torch.cat([pad_emb, emb], 0)

        self.embedding_cache[user_id] = emb_table
        return emb_table
