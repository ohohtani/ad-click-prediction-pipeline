import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dl_common import *


def train_model(train_df, num_cols, cat_cols, seq_col, target_col, batch_size, epochs, lr, device, cat_cardinalities):
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train, pin_memory=True)
    model = WideDeepCTR(num_features=len(num_cols), cat_cardinalities=cat_cardinalities, emb_dim=16, lstm_hidden=64, hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]).to(device)

    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=2, T_mult=2)

    print('학습 시작')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for num_x, cat_x, seqs, lens, ys in tqdm(train_loader, desc=f'[Train Epoch {epoch}]'):
            num_x, cat_x, seqs, lens, ys = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device), ys.to(device)
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(num_x, cat_x, seqs, lens), ys).backward()
            optimizer.second_step(zero_grad=True)
            total_loss += loss.item() * ys.size(0)
        scheduler.step()
        total_loss /= len(train_dataset)
        print(f'[Epoch {epoch}] Train Loss: {total_loss:.4f}')
    print('학습 완료')
    return model


def run_training():
    ensure_directories()
    seed_everything(CFG['SEED'])
    device = get_device()
    print(f'Device: {device}')
    print('데이터 로드 시작')
    train = pd.read_parquet(TRAIN_PATH, engine='pyarrow')
    test = pd.read_parquet(TEST_PATH, engine='pyarrow')
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    print('데이터 로드 완료')

    _, num_cols = get_feature_lists(train)
    print(f'Num features: {len(num_cols)} | Cat features: {len(CAT_COLS)}')
    train, test, encoders = encode_categoricals(train, test, CAT_COLS)
    cat_cardinalities = [len(encoders[c]) for c in CAT_COLS]
    save_json(ENCODER_PATH, encoders)
    save_json(META_PATH, {'cat_cols': CAT_COLS, 'num_cols': num_cols, 'seq_col': SEQ_COL, 'target_col': TARGET_COL, 'n_folds': CFG['N_FOLDS'], 'batch_size': CFG['BATCH_SIZE'], 'cat_cardinalities': cat_cardinalities})

    kf = KFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])
    for fold, (train_idx, _) in enumerate(kf.split(train), start=1):
        print(f'\n========== Fold {fold} / {CFG["N_FOLDS"]} ==========')
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        model = train_model(train_fold, num_cols, CAT_COLS, SEQ_COL, TARGET_COL, CFG['BATCH_SIZE'], CFG['EPOCHS'], CFG['LEARNING_RATE'], device, cat_cardinalities)
        save_path = os.path.join(MODELS_DIR, f'wide_deep_fold{fold}.pt')
        torch.save(model.state_dict(), save_path)
        print(f'[Fold {fold}] 모델 가중치 저장 완료 → {save_path}')

    print('\n✅ 모든 Fold 학습 및 모델 저장 완료.')
