import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dl_common import *


def run_inference():
    device = get_device()
    print(f'Device: {device}')
    test = pd.read_parquet(TEST_PATH, engine='pyarrow')
    sample = pd.read_csv(SAMPLE_SUB)
    meta = load_json(META_PATH)
    encoders = load_json(ENCODER_PATH)
    cat_cols = meta['cat_cols']
    num_cols = meta['num_cols']
    seq_col = meta['seq_col']
    n_folds = meta['n_folds']
    cat_cardinalities = meta['cat_cardinalities']

    test = apply_saved_encoders(test, encoders, cat_cols)
    test_dataset = ClickDataset(test, num_cols, cat_cols, seq_col, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_infer, pin_memory=True)

    all_preds = []
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(MODELS_DIR, f'wide_deep_fold{fold}.pt')
        print(f'\n[Fold {fold}] 모델 로드 중: {model_path}')
        model = WideDeepCTR(num_features=len(num_cols), cat_cardinalities=cat_cardinalities, emb_dim=16, lstm_hidden=64, hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        preds = []
        with torch.no_grad():
            for num_x, cat_x, seqs, lens in tqdm(test_loader, desc=f'[Inference Fold {fold}]'):
                num_x, cat_x, seqs, lens = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device)
                logits = model(num_x, cat_x, seqs, lens)
                preds.append(torch.sigmoid(logits).cpu())
        all_preds.append(torch.cat(preds).numpy())

    final_preds = np.mean(all_preds, axis=0)
    sample['clicked'] = final_preds
    sample.to_csv(DL_OUT_PATH, index=False)
    print(f'\n✅ 추론 완료 — 결과 저장: {DL_OUT_PATH}')
