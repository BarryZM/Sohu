import os

OUTPUT_DIR = './model/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import pandas as pd
import numpy as np
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings

warnings.filterwarnings("ignore")

import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers

print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

transformers.logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import json
import warnings

warnings.filterwarnings('ignore')


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


# =======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)

class CFG:
    apex=True
    num_workers=8
    model="hfl/chinese-pert-large"
    scheduler='cosine'
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=3
    last_epoch=-1
    encoder_lr=1e-5
    decoder_lr=1e-5
    batch_size=4
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    seed=2022
    n_fold= 5
    trn_fold=[0,1,2,3, 4]
    train=True


train = pd.read_json('Sohu2022_data/train.txt', lines=True)
temp_fu = pd.read_json('Sohu2022_data/nlp-train-dataset.txt', lines=True)
train = pd.concat([train, temp_fu], axis=0, sort=False).reset_index(drop=True)

def cut_sentences(content):
    sentences = re.split(r'。|！|？', content)
    return sentences

def process_text(content, entity, front_entity):

    sentence_list = cut_sentences(content)
    content1 = '。'.join([i for i in sentence_list if front_entity in i])
    content2 = '。'.join([i for i in sentence_list if front_entity not in i])
    return front_entity + '_' + ','.join([i for i in entity]) + '_' + content1 + '_' + content2

def get_single_mode(df):
    base_id = list(df['id'])
    base_content = list(df['content'])
    base_entity = list(df['entity'])

    entity = []
    label = []
    content = []
    id_list = []

    for i in tqdm(range(len(base_id))):
        for k in base_entity[i]:
            id_list.append(base_id[i])
            entity.append(k)
            temp_label = base_entity[i][k] + 2
            label.append(temp_label)
            content.append(process_text(base_content[i], base_entity[i], k))
    new_df = pd.DataFrame()
    new_df['id'] = id_list
    new_df['content'] = content
    new_df['entity'] = entity
    new_df['label'] = label
    return new_df


train = get_single_mode(train)

Fold = GroupKFold(n_splits=CFG.n_fold)
groups = train['id'].values
for n, (train_index, val_index) in enumerate(Fold.split(train, train['label'], groups)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer

def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           truncation = True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.contents[item],
                               self.entitys[item])
        labels = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 5)
        self._init_weights(self.fc)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = torch.mean(outputs[0], axis=1)
        return last_hidden_states

    def loss(self, logits, labels):
        # loss_fnc = nn.CrossEntropyLoss()
        loss_fnc = FocalLoss()
        loss = loss_fnc(logits, labels)
        return loss

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        logits1 = self.fc(self.drop1(feature))
        logits2 = self.fc(self.drop2(feature))
        logits3 = self.fc(self.drop3(feature))
        logits4 = self.fc(self.drop4(feature))
        logits5 = self.fc(self.drop5(feature))
        output = self.fc(feature)
        output = F.softmax(output, dim=1)
        _loss = 0
        if labels is not None:
            loss1 = self.loss(logits1, labels)
            loss2 = self.loss(logits2, labels)
            loss3 = self.loss(logits3, labels)
            loss4 = self.loss(logits4, labels)
            loss5 = self.loss(logits5, labels)
            _loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        return output, _loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_fn(fold, train_loader,model, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    # start = end = time.time()
    global_step = 0
    grad_norm = 0
    tk0=tqdm(enumerate(train_loader),total=len(train_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds,loss = model(inputs,labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        tk0.set_postfix(Epoch=epoch+1, Loss=losses.avg,lr=scheduler.get_lr()[0])
    return losses.avg

def valid_fn(valid_loader, model, device):
    losses = AverageMeter()
    model.eval()
    # preds = []
    valid_true = []
    valid_pred = []
    tk0=tqdm(enumerate(valid_loader),total=len(valid_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds,loss = model(inputs,labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        losses.update(loss.item(), batch_size)
        batch_pred = y_preds.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(-1))
        for item in np.array(labels.cpu()):
            valid_true.append(item)
        tk0.set_postfix(Loss=losses.avg)
    print('Test set: Average loss: {:.4f}'.format(losses.avg))
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    print('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    print(classification_report(valid_true, valid_pred))

    return avg_acc, avg_f1s, losses.avg


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds.copy()
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    print(len(train_loader), len(valid_loader))

    # ====================================================
    # model & optimizer
    # ====================================================
    best_score = 0.
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR + 'config.pth')

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr': encoder_lr},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr': encoder_lr},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr': decoder_lr}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles, last_epoch=((cfg.last_epoch + 1) / cfg.epochs) * num_train_steps
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    if torch.cuda.device_count() > 1:
        print("Currently training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # ====================================================
    # loop
    # ====================================================

    for epoch in range(CFG.epochs - 1 - CFG.last_epoch):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, optimizer, epoch, scheduler, device)

        # eval
        avg_acc, avg_f1s, valid_loss = valid_fn(valid_loader, model, device)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f} time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {avg_f1s:.4f}')

        if best_score < avg_f1s:
            best_score = avg_f1s
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: f1: {best_score:.4f} Model')
        torch.save(model.state_dict(), OUTPUT_DIR + f"model_fold{fold}_best.bin")

    torch.cuda.empty_cache()
    gc.collect()

if CFG.train:
    train_loop(train, fold=0)


############ test
test = pd.read_json('data/nlp-test-dataset.txt', lines=True)
def get_new_sentence(row):
    new_dict = {}
    for i in row:
        new_dict[i] = 0
    return new_dict
test['entity'] = test['entity'].apply(lambda row:get_new_sentence(row))
test = get_single_mode(test)
test_ids = test['id'].values
test_entitys = test['entity'].values

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.contents[item],
                               self.entitys[item])
        return inputs

def test_and_save_reault(device, test_loader, test_ids, result_path):
    raw_preds = []
    test_pred = []
    for fold in range(1):
        current_idx = 0

        model = CustomModel(CFG, config_path=OUTPUT_DIR + 'config.pth', pretrained=True)
        model.to('cuda')
        model.load_state_dict(
            torch.load(os.path.join(OUTPUT_DIR, f"model_fold{fold}_best.bin"), map_location=torch.device('cuda')))
        model.eval()
        tk0 = tqdm(test_loader, total=len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_pred_pa_all, _ = model(inputs)
            batch_pred = (y_pred_pa_all.detach().cpu().numpy()) / CFG.n_fold
            if fold == 0:
                raw_preds.append(batch_pred)
            else:
                raw_preds[current_idx] += batch_pred
                current_idx += 1

    np.save(OUTPUT_DIR + 'test.npy', raw_preds)

    for preds in raw_preds:
        for item in preds:
            test_pred.append(item.argmax(-1))
    assert len(test_entitys) == len(test_pred) == len(test_ids)
    result = {}
    for id, entity, pre_lable in zip(test_ids, test_entitys, test_pred):

        pre_lable = int(pre_lable) - 2
        if id in result.keys():
            result[id][entity] = pre_lable
        else:
            result[id] = {entity: pre_lable}
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("id	result")
        f.write('\n')
        for k, v in result.items():
            f.write(str(k) + '	' + json.dumps(v, ensure_ascii=False) + '\n')
    print(f"保存文件到:{result_path}")


test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                  batch_size=16,
                  shuffle=False,
                  num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

test_and_save_reault(device, test_loader, test_ids, OUTPUT_DIR+'output.txt')

val_pred = np.load(OUTPUT_DIR + 'test.npy', allow_pickle=True)
val_pred = np.concatenate(val_pred)

weights = search_weight(val_pred, val_pred)
weights = [4.21, 0.88, 0.03, 0.24, 3.22]
val_pred = val_pred * np.array(weights)

res = pd.DataFrame()
res['id'] = test_ids
res['entity'] = test_entitys
res['label'] = np.argmax(val_pred, axis=1) - 2
res = res.groupby(['id']).apply(lambda row: dict(zip(list(row['entity']), list(row['label'])))).reset_index()
res.columns = ['id', 'result']
res.to_csv('section1.txt', index=False, sep='\t')





