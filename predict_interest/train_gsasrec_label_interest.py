from argparse import ArgumentParser
import os

import torch
import time
from utils import load_config, build_model_label, get_device
from dataset_utils import get_dataloader, get_num_items
from data_iterator_label import DataIteratorLabel
from tqdm import tqdm
from eval_utils import evaluate
from torchinfo import summary

print("train_gsasrec_for_ver3.py", flush=True)

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
    
print("-------------", flush=True)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)

print("-------------", flush=True)

num_items = config.num_items
device = get_device()
model = build_model_label(config, config.num_items)

# ここが要変更
# train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
#                                          max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
# val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)

print('dataloade making begin', flush=True)

train_dataloader = DataIteratorLabel(config.train_file_path, config.train_batch_size, config.sequence_length, train_flag=0, padding_value=num_items+1, train_neg_per_pos=config.negs_per_pos)
val_dataloader = get_dataloader(input_path=config.valid_input_path, output_path=config.valid_output_path, batch_size=config.train_batch_size, max_length=config.sequence_length, padding_value=num_items+1)

print('dataloder making end', flush=True)

optimiser = torch.optim.Adam(model.parameters())
# batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))
test_iter = round(config.num_user / config.train_batch_size)

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0
loss_sum = 0

model = model.to(device)
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

model.train()   

start_time = time.time()

print('training_begin, test_iter: ', test_iter, flush=True)

for positives, negatives in train_dataloader:
    step += 1
    positives = positives.to(device)
    negatives = negatives.to(device)
    model_input = positives[:, :-1]
    last_hidden_state, attentions = model(model_input)
    labels = positives[:, 1:]
    negatives = negatives[:, 1:, :]
    pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
    output_embeddings = model.get_output_embeddings()
    pos_neg_embeddings = output_embeddings(pos_neg_concat)
    mask = (model_input != num_items + 1).float()
    logits = torch.einsum('bse, bsne -> bsn', last_hidden_state, pos_neg_embeddings)
    gt = torch.zeros_like(logits)
    gt[:, :, 0] = 1

    alpha = config.negs_per_pos / (num_items - 1)
    t = config.gbce_t 
    beta = alpha * ((1 - 1/alpha)*t + 1/alpha)
    
    positive_logits = logits[:, :, 0:1].to(torch.float64) #use float64 to increase numerical stability
    negative_logits = logits[:,:,1:].to(torch.float64)
    eps = 1e-10
    positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1-eps)
    positive_probs_adjusted = torch.clamp(positive_probs.pow(-beta), 1+eps, torch.finfo(torch.float64).max)
    to_log = torch.clamp(torch.div(1.0, (positive_probs_adjusted  - 1)), eps, torch.finfo(torch.float64).max)
    positive_logits_transformed = to_log.log()
    logits = torch.cat([positive_logits_transformed, negative_logits], -1)
    loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1)*mask
    loss = loss_per_element.sum() / mask.sum()
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    loss_sum += loss.item()
    
    if(step%1000==0):
        print(f"step: {step}", flush=True)

    if step % test_iter == 0: # 終了条件チェックをtest_iterごとに行う
        evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                    config.filter_rated, device=device) 
        log_str = 'iter: %d, train loss: %.4f' % (step, loss_sum / test_iter)
        log_str += ', ' + str(config.val_metric) + ': %.6f' % (evaluation_result[config.val_metric])
        test_time = time.time()
        print(log_str, flush=True)
        print("time interval: %.4f min" % ((test_time-start_time)/60.0), flush=True)
        loss_sum = 0
        
        if evaluation_result[config.val_metric] > best_metric:
            best_metric = evaluation_result[config.val_metric]
            model_name = f"models_label/gsasrec-{config.dataset_name}-step:{step}-t:{config.gbce_t}-negs:{config.negs_per_pos}-emb:{config.embedding_dim}-dropout:{config.dropout_rate}-metric:{best_metric}.pt" 
            print(f"Saving new best model to {model_name}")
            if best_model_name is not None:
                os.remove(best_model_name)
            best_model_name = model_name
            steps_not_improved = 0
            torch.save(model.state_dict(), model_name)
        else:
            steps_not_improved += 1
            print(f"Validation metric did not improve for {steps_not_improved} steps")
            if steps_not_improved >= config.early_stopping_patience:
                print(f"Stopping training, best model was saved to {best_model_name}")
                break
            
    if step/test_iter > config.max_epochs:
        print(f"Stopping training, best model was saved to {best_model_name}")
        break
        