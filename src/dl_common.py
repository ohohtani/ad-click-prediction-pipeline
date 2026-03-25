import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

DATA_DIR = './data'
MODELS_DIR = './models/wide_deep'
OUTPUT_DIR = './outputs'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.parquet')
TEST_PATH = os.path.join(DATA_DIR, 'test.parquet')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')
DL_OUT_PATH = os.path.join(OUTPUT_DIR, 'wide_deep_inference.csv')
META_PATH = os.path.join(MODELS_DIR, 'wide_deep_meta.json')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoders.json')

CFG = {'BATCH_SIZE': 1024, 'EPOCHS': 5, 'LEARNING_RATE': 1e-3, 'SEED': 42, 'N_FOLDS': 5}
CAT_COLS = ['gender', 'age_group', 'inventory_id', 'l_feat_14']
TARGET_COL = 'clicked'
SEQ_COL = 'seq'
FEATURE_EXCLUDE = {TARGET_COL, SEQ_COL, 'ID'}


def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_feature_lists(df):
    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    num_cols = [c for c in feature_cols if c not in CAT_COLS]
    return feature_cols, num_cols


def encode_categoricals(train_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna('UNK')
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna('UNK'))
        test_df[col] = le.transform(test_df[col].astype(str).fillna('UNK'))
        encoders[col] = list(le.classes_)
        print(f'{col} unique categories: {len(le.classes_)}')
    return train_df, test_df, encoders


def apply_saved_encoders(df, encoders, cat_cols):
    df = df.copy()
    for col in cat_cols:
        classes = encoders[col]
        mapping = {v: i for i, v in enumerate(classes)}
        unk_idx = mapping.get('UNK', 0)
        df[col] = df[col].astype(str).fillna('UNK').map(lambda x: mapping.get(x, unk_idx)).astype(int)
    return df


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        arr = np.fromstring(s, sep=',', dtype=np.float32) if s else np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        return num_x, cat_x, seq


def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys


def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x


class WideDeepCTR(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64, hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities])
        cat_input_dim = emb_dim * len(cat_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        seq_out_dim = lstm_hidden * 2
        self.cross = CrossNetwork(num_features + cat_input_dim + seq_out_dim, num_layers=2)
        input_dim = num_features + cat_input_dim + seq_out_dim
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout[i % len(dropout)])]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        num_x = self.bn_num(num_x)
        cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
        cat_feat = torch.cat(cat_embs, dim=1)
        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(seqs, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        z = torch.cat([num_x, cat_feat, h], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)


import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

