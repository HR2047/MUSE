import json
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, rated_file=None, max_length=200):
        with open(input_file, 'r') as f:
            self.inputs = [list(map(int, line.strip().split())) for line in f.readlines()]

        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [int(line.strip()) for line in f.readlines()]
        else:
            self.outputs = None

        if rated_file:
            with open(rated_file, 'r') as f:
                self.rated = [set(map(int, line.strip().split())) for line in f.readlines()]
        else:
            self.rated = [set(seq) for seq in self.inputs]  # fallback: rated = input のまま

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        rated = self.rated[idx]

        if len(inp) > self.max_length:
            inp = inp[-self.max_length:]
        elif len(inp) < self.max_length:
            inp = [self.padding_value] * (self.max_length - len(inp)) + inp  # 左パディング

        inp_tensor = torch.tensor(inp, dtype=torch.long)

        if self.outputs:
            out_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return inp_tensor, rated, out_tensor 

        return inp_tensor,

def collate_with_random_negatives(input_batch, pad_value, num_negatives):
    batch_cat = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    negatives = torch.randint(low=1, high=pad_value, size=(batch_cat.size(0), batch_cat.size(1), num_negatives))
    return [batch_cat, negatives]

def collate_val_test(input_batch):
    input = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    rated = [input_batch[i][1] for i in range(len(input_batch))]
    output = torch.stack([input_batch[i][2] for i in range(len(input_batch))], dim=0)
    return [input, rated, output]

def get_num_items(dataset):
    with open(f"../data/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']

def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    padding_value = stats['num_items'] + 1
    return padding_value

def get_val_or_test_dataloader(padding_value, input_file_path, output_file_path, rated_file_path=None, batch_size=32, max_length=20):
    dataset = SequenceDataset(input_file_path, padding_value, output_file_path, rated_file=rated_file_path, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test, drop_last=True)
    return dataloader

def get_test_dataloader_gsasrec(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"../data/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    dataset = SequenceDataset(f"{dataset_dir}/{part}/input.txt", padding_value,  f"{dataset_dir}/{part}/output.txt", max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return dataloader

def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_test_dataloader_gsasrec(dataset_name, 'test', batch_size, max_length)

def get_hist_mask(padding_value, hist_item):
    return [[0.0 if value == padding_value else 1.0 for value in row] for row in hist_item]

def get_dataloader(input_path, output_path, batch_size, max_length, padding_value):
    dataset = SequenceDataset(input_path, padding_value, output_path, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return dataloader

class TestDataset(Dataset):
    def __init__(self, data_path, target_path, transform=None):
        with open(data_path, 'r') as f:
            self.data = [float(line.strip()) for line in f]

        with open(target_path, 'r') as f:
            self.targets = [int(line.strip()) for line in f]

        assert len(self.data) == len(self.targets), "Data and target lengths do not match"
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)