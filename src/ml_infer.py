import gc
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool

from .ml_common import *

warnings.filterwarnings('ignore')

def predict_with_single_model(test_path, usecols, kept, base_cat_cols, enc_json_path, cbm_path, enable_history_aggs):
    # 인코딩맵 로드
    with open(enc_json_path, 'r', encoding='utf-8') as f:
        enc_maps = json.load(f)
    woe_maps  = enc_maps.get('woe', {})
    te_maps   = enc_maps.get('te', {})
    freq_maps = enc_maps.get('freq', {})

    # 모델 로드
    model = CatBoostClassifier()
    model.load_model(cbm_path)

    # row-group 단위로 누적 예측
    pf = pq.ParquetFile(test_path)
    pred_parts = []
    rows_total = 0

    for rg in range(pf.num_row_groups):
        cols_avail = [c for c in usecols if c in pf.schema.names]
        tbl = pf.read_row_group(rg, columns=cols_avail)
        pdf = tbl.to_pandas()
        rows_total += len(pdf)

        proc = create_revolutionary_features(pdf, enable_history_aggs=enable_history_aggs, drop_seq_after=True)

        # 인코딩 (해당 모델 전용 맵)
        proc = apply_woe_encoding(proc, woe_maps)
        proc = apply_target_encoding(proc, te_maps)
        proc = apply_freq_maps(proc, freq_maps)

        # kept 순서로 정렬 + 결측/카테고리 처리
        for c in kept:
            if c not in proc.columns:
                proc[c] = np.nan

        X = proc[kept].copy()
        cat_cols2 = [c for c in base_cat_cols if c in kept and c in X.columns]
        X = sanitize_categoricals(X, cat_cols2)
        num_cols = [c for c in kept if c not in cat_cols2]
        for c in num_cols:
            if X[c].dtype.kind in 'biufc':
                med = X[c].median()
                X[c] = X[c].fillna(med)

        pool = Pool(X, cat_features=cat_cols2)
        pred = model.predict_proba(pool)[:, 1].astype('float32')
        pred_parts.append(pred)

        del tbl, pdf, proc, X, pool, pred
        gc.collect()

    preds = np.concatenate(pred_parts, axis=0) if pred_parts else np.array([], dtype=np.float32)
    return preds

def run_inference():
    # 모델/맵/kept 로딩
    pairs = list_models_and_maps(CB_DIR, ENC_DIR)     # [(cbm, json), ...] 50쌍
    kept = load_kept_features(KEPT_PATH)

    # 테스트에 history_*가 존재하면 history 집계 사용
    test_raw_cols = get_raw_columns(TEST_PATH)
    enable_history_aggs = any(str(c).startswith('history_') for c in test_raw_cols)
    print(f"[feature-compat] ENABLE_HISTORY_AGGS={enable_history_aggs}")

    base_cat_cols = list(BASE_CAT_COLS)

    # 각 모델별 예측 → 동일가중 평균
    n_models = len(pairs)
    print(f"[load] models={n_models}")
    sum_pred = None

    for i, (cbm_path, enc_path) in enumerate(pairs, start=1):
        print(f"\n[{i}/{n_models}] Predict with {os.path.basename(cbm_path)} | enc={os.path.basename(enc_path)}")
        preds = predict_with_single_model(
            TEST_PATH, USECOLS, kept, base_cat_cols, enc_path, cbm_path, enable_history_aggs
        )
        if sum_pred is None:
            sum_pred = preds.astype('float64')
        else:
            # 길이 안전 장치(불일치 시 최소 길이로 절단)
            m = min(len(sum_pred), len(preds))
            sum_pred = sum_pred[:m] + preds[:m]

        gc.collect()

    if sum_pred is None:
        raise RuntimeError("예측 결과가 없습니다.")
    p_final = (sum_pred / n_models).astype('float32')
    print(f"\n[Final] equal-avg over {n_models} models → mean={p_final.mean():.6f} min={p_final.min():.6f} max={p_final.max():.6f}")

    # 제출 파일 만들기
    sub = pd.read_csv(SAMPLE_SUB)
    n = min(len(sub), len(p_final))
    if len(sub) != len(p_final):
        print(f"[경고] 길이 불일치: sample={len(sub)}, pred={len(p_final)} → min={n}로 절단")
    sub = sub.iloc[:n].copy()
    label_col = sub.columns[1] if len(sub.columns) >= 2 else 'clicked'
    sub[label_col] = p_final[:n]
    sub.to_csv(OUT_PATH, index=False, float_format='%.8f')
    print(f"[OK] 저장: {OUT_PATH}")

