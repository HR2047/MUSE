from argparse import ArgumentParser

import torch
import sys
from dataset_utils import get_num_items, get_test_dataloader, get_dataloader

sys.path.append("../predict_interest")
from eval_utils import evaluate
from utils import build_model_label, get_device, load_config

import os
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

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--result_csv_path', default="log/result.csv")
parser.add_argument('--exp_name', default="muse_label")   
args = parser.parse_args()
config = load_config(args.config)
device = get_device()
model = build_model_label(config, num_items=config.num_items)
model = model.to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))

# test_dataloader = get_test_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)
test_dataloader = get_dataloader(input_path=config.test_input_path, output_path=config.test_output_path, batch_size=config.eval_batch_size, max_length=config.sequence_length, padding_value=config.num_items+1)
evaluation_result = evaluate(model, test_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device) 
print(evaluation_result)
save_result_to_csv(args.result_csv_path, args.exp_name, evaluation_result)

