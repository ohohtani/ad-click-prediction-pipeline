import gc
import glob
import os
import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

# Paths
DATA_DIR = './data'
MODELS_DIR = './models/ml_ctr'
OUTPUT_DIR = './outputs'
REPORT_DIR = './reports/ml_ctr'
FEATURE_IMPORTANCE_DIR = './reports/ml_ctr_feature_importance'
STAGING_DIR = './artifacts/neg_bags_10bag_ratio1.0'
CB_DIR = os.path.join(MODELS_DIR, 'catboost')
ENC_DIR = os.path.join(MODELS_DIR, 'enc_maps')
KEPT_PATH = os.path.join(MODELS_DIR, 'kept_features.txt')
META_PATH = os.path.join(MODELS_DIR, 'meta.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.parquet')
TEST_PATH = os.path.join(DATA_DIR, 'test.parquet')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')
OUT_PATH = os.path.join(OUTPUT_DIR, 'final_submission.csv')

GLOBAL_SEED = 42
N_BAGS = 10
NEG_RATIO = 1.0
RANDOM_SEED = 42
N_FOLDS = 5
FOLD_SHUFFLE = True
FOLD_RANDOM_SEED = 2025
N_PRUNE_BOTTOM = 10

CB_PARAMS = dict(
    iterations=8000,
    learning_rate=0.012,
    depth=10,
    l2_leaf_reg=6,
    bootstrap_type='Bernoulli',
    subsample=0.75,
    border_count=254,
    feature_border_type='GreedyLogSum',
    min_data_in_leaf=25,
    random_strength=2.5,
    loss_function='Logloss',
    eval_metric='AUC',
    custom_metric=['Precision','Recall','F1'],
    one_hot_max_size=20,
    max_ctr_complexity=4,
    verbose=100,
)

USECOLS = ['clicked', 'seq', 'hour', 'day_of_week', 'inventory_id', 'gender', 'age_group']
USECOLS += [f'l_feat_{i}' for i in range(1, 28)]
USECOLS += [f'feat_a_{i}' for i in range(1, 19)]
USECOLS += [f'feat_b_{i}' for i in range(1, 7)]
USECOLS += [f'feat_c_{i}' for i in range(1, 9)]
USECOLS += [f'feat_d_{i}' for i in range(1, 7)]
USECOLS += [f'feat_e_{i}' for i in range(1, 11)]
USECOLS += [f'history_a_{i}' for i in range(1, 8)]
USECOLS += [f'history_b_{i}' for i in range(1, 31)]

WOE_TARGET_COLS = [
    'inventory_id', 'seq_last', 'seq_first', 'seq_most_common',
    'x_inv__hourbin', 'x_inv__dow', 'x_age__hourbin',
    'x_invGroup__age', 'x_inv__gender', 'x_inv__age',
    'diversity_bin', 'seq_pattern_type', 'hour_group'
]
TE_TARGET_COLS = WOE_TARGET_COLS.copy()
FREQ_TARGET_COLS = WOE_TARGET_COLS + [
    'seq_second_common', 'dow_hour_interaction',
    'x_seqcommon__inv', 'seq_first_last_pair'
]
BASE_CAT_COLS = [
    'age_group','inventory_id','inventory_group','hour_group','gender',
    'seq_first','seq_last','seq_most_common','seq_second_common','seq_pattern_type',
    'x_inv__hourbin','x_inv__gender','x_inv__age','x_age__hourbin',
    'x_inv__dow','x_age__dow','x_invGroup__age','x_gender__hourbin','x_gender__dow',
    'x_age__gender','x_invGroup__hourgroup','x_invGroup__dow','x_hourgroup__dow',
    'diversity_bin','dow_hour_interaction','x_seqcommon__inv','seq_first_last_pair','seq_len_bin'
]


def ensure_directories():
    for path in [MODELS_DIR, OUTPUT_DIR, REPORT_DIR, FEATURE_IMPORTANCE_DIR, STAGING_DIR, CB_DIR, ENC_DIR]:
        os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)


def calculate_competition_score(y_true, y_pred_proba):
    ap = average_precision_score(y_true, y_pred_proba)
    eps = 1e-15
    p = np.clip(y_pred_proba, eps, 1 - eps)
    pos = (y_true == 1); neg = ~pos
    pos_ll = -np.mean(np.log(p[pos])) if pos.any() else 0.0
    neg_ll = -np.mean(np.log(1 - p[neg])) if neg.any() else 0.0
    wll = 0.5 * pos_ll + 0.5 * neg_ll
    final = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return {'final_score': float(final), 'ap': float(ap), 'wll': float(wll)}


def sanitize_categoricals(df, cat_cols):
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('object').where(~pd.isna(df[c]), 'NA').astype(str)
    return df


def get_raw_columns(path):
    pf = pq.ParquetFile(path)
    return set(pf.schema.names)


def count_pos_neg(train_path):
    pf = pq.ParquetFile(train_path)
    total_neg, total_pos = 0, 0
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=['clicked'])
        col = tbl.column('clicked').to_numpy(zero_copy_only=False)
        total_neg += (col == 0).sum()
        total_pos += (col == 1).sum()
        del tbl, col
    return total_pos, total_neg


def materialize_pos_and_neg_bags(train_path, usecols, total_pos, total_neg, per_bag_neg, n_bags, seed, staging_dir):
    os.makedirs(staging_dir, exist_ok=True)
    pos_path = os.path.join(staging_dir, 'pos_all.parquet')
    neg_paths = [os.path.join(staging_dir, f'neg_bag{i+1}.parquet') for i in range(n_bags)]
    if os.path.exists(pos_path) and all(os.path.exists(p) for p in neg_paths):
        print('[staging] 기존 분할 파일을 사용합니다.')
        return pos_path, neg_paths

    use_bags = min(n_bags, total_neg // max(1, per_bag_neg))
    if use_bags < n_bags:
        print(f'[경고] 요청 bag={n_bags} 중 {use_bags}개만 생성 가능(음성 부족).')
        neg_paths = neg_paths[:use_bags]
        n_bags = use_bags

    print(f'[staging] 디스조인트 음성 분할 저장 중… (bags={n_bags}, per_bag_neg={per_bag_neg:,})')
    rng = np.random.default_rng(seed)
    choose = rng.permutation(total_neg)[:per_bag_neg * n_bags]
    assignments = np.full(total_neg, -1, dtype=np.int32)
    for b in range(n_bags):
        sl = choose[b*per_bag_neg:(b+1)*per_bag_neg]
        assignments[sl] = b

    pf = pq.ParquetFile(train_path)
    pos_writer = None
    neg_writers = [None] * n_bags
    neg_seen = 0
    for rg in range(pf.num_row_groups):
        cols_avail = [c for c in usecols if c in pf.schema.names]
        tbl = pf.read_row_group(rg, columns=cols_avail)
        pdf = tbl.to_pandas()

        pos_mask = (pdf['clicked'] == 1)
        if pos_mask.any():
            t = pa.Table.from_pandas(pdf.loc[pos_mask], preserve_index=False)
            if pos_writer is None:
                pos_writer = pq.ParquetWriter(pos_path, t.schema)
            pos_writer.write_table(t)

        neg_mask = (pdf['clicked'] == 0)
        if neg_mask.any():
            idx_neg = np.flatnonzero(neg_mask)
            g_idx = neg_seen + np.arange(len(idx_neg), dtype=np.int64)
            bag_ids = assignments[g_idx]
            m = (bag_ids >= 0)
            if np.any(m):
                sel_idx = idx_neg[m]
                sel_bag = bag_ids[m]
                for bag_id in np.unique(sel_bag):
                    loc = sel_idx[sel_bag == bag_id]
                    if len(loc) == 0:
                        continue
                    t = pa.Table.from_pandas(pdf.iloc[loc], preserve_index=False)
                    if neg_writers[bag_id] is None:
                        neg_writers[bag_id] = pq.ParquetWriter(neg_paths[bag_id], t.schema)
                    neg_writers[bag_id].write_table(t)
            neg_seen += len(idx_neg)

        del tbl, pdf
        gc.collect()

    if pos_writer is not None:
        pos_writer.close()
    for w in neg_writers:
        if w is not None:
            w.close()
    print('[staging] 분할 저장 완료.')
    return pos_path, neg_paths


def nat_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


def list_models_and_maps(cb_dir, enc_dir):
    cbms = sorted(glob.glob(os.path.join(cb_dir, 'b??_f??.cbm')), key=nat_key)
    pairs = []
    for m in cbms:
        base = os.path.splitext(os.path.basename(m))[0]
        enc = os.path.join(enc_dir, f'{base}.json')
        if not os.path.exists(enc):
            raise FileNotFoundError(f'인코딩 맵 누락: {enc}')
        pairs.append((m, enc))
    if len(pairs) == 0:
        raise FileNotFoundError(f'모델(.cbm)이 {cb_dir}에 없음')
    return pairs


def load_kept_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'kept_features.txt 누락: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        kept = [ln.strip() for ln in f if ln.strip()]
    if len(kept) == 0:
        raise ValueError('kept_features.txt가 비어있습니다.')
    return kept

def _to_str(sr: pd.Series) -> pd.Series:
    return sr.astype('object').where(~pd.isna(sr), 'NA').astype(str)

def _cross_str(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = _to_str(a); bb = _to_str(b)
    return aa.str.cat(bb, sep='|').astype('category')

def _seq_unique_count(seq_str: str) -> int:
    toks = seq_str.split(',') if isinstance(seq_str, str) else []
    return len(set(toks))

def _seq_entropy(seq_str: str) -> float:
    if not isinstance(seq_str, str) or not seq_str:
        return 0.0
    toks = seq_str.split(',')
    if not toks: return 0.0
    cnt = Counter(toks); n = float(len(toks))
    ps = [c / n for c in cnt.values()]
    return float(-np.sum([p * np.log(p + 1e-12) for p in ps]))

def _seq_most_common(seq_str):
    if not isinstance(seq_str, str): return 'NA'
    tokens = seq_str.split(',')
    if not tokens: return 'NA'
    cnt = Counter(tokens)
    return cnt.most_common(1)[0][0] if cnt else 'NA'

def _seq_second_common(seq_str):
    if not isinstance(seq_str, str): return 'NA'
    tokens = seq_str.split(',')
    if len(tokens) <= 1: return 'NA'
    cnt = Counter(tokens).most_common(2)
    return cnt[1][0] if len(cnt) >= 2 else 'NA'

def _seq_diversity(seq_str):
    if not isinstance(seq_str, str): return 0.5
    tokens = seq_str.split(',')
    if not tokens: return 0.5
    return len(set(tokens)) / max(1, len(tokens))

def _seq_max_repetition(seq_str):
    if not isinstance(seq_str, str): return 1
    tokens = seq_str.split(',')
    if not tokens: return 1
    cnt = Counter(tokens)
    return max(cnt.values()) if cnt else 1

def _seq_pattern_type(seq_str):
    if not isinstance(seq_str, str): return 'unknown'
    tokens = seq_str.split(',')
    if len(tokens) < 2: return 'too_short'
    start, end = tokens[0], tokens[-1]
    if start == end: return 'circular'
    key_items = {'74', '269', '479', '88', '92'}
    if start in key_items and end in key_items: return 'key_to_key'
    elif start in key_items: return 'key_start'
    elif end in key_items: return 'key_end'
    return 'other'

def create_revolutionary_features(df, enable_history_aggs=True, drop_seq_after=True):
    df = df.copy()

    # 기본 시퀀스 파생
    s = df['seq'].astype(str)
    df['seq_length'] = s.str.count(',').astype('int16') + 1
    df['seq_first']  = s.str.split(',', n=1).str[0].astype('category')
    df['seq_last']   = s.str.rsplit(',', n=1).str[-1].astype('category')

    df['seq_unique_count'] = s.apply(_seq_unique_count).astype('int16')
    df['seq_diversity']    = s.apply(_seq_diversity).astype('float32')
    df['seq_entropy']      = s.apply(_seq_entropy).astype('float32')
    df['seq_most_common']  = s.apply(_seq_most_common).astype('category')
    df['seq_second_common']= s.apply(_seq_second_common).astype('category')
    df['seq_max_repetition']= s.apply(_seq_max_repetition).astype('int8')
    df['seq_pattern_type'] = s.apply(_seq_pattern_type).astype('category')

    df['seq_first_last_pair'] = (df['seq_first'].astype(str) + '_' +
                                 df['seq_last'].astype(str)).astype('category')

    important_items = ['74', '269', '479', '88', '92', '21', '41', '37', '91']
    for item in important_items:
        df[f'seq_count_{item}'] = s.str.count(item).astype('int8')
    df['seq_total_key_counts'] = sum([df[f'seq_count_{item}'] for item in ['74', '269', '479']]).astype('int16')

    # 경량 추가 파생
    key_all = ['74','269','479','88','92','21','41','37','91']
    df['seq_total_key_counts_ext'] = sum([df.get(f'seq_count_{k}', 0) for k in key_all]).astype('int16')
    df['seq_key_prop']     = (df['seq_total_key_counts_ext'] / df['seq_length'].clip(lower=1)).astype('float32')
    df['seq_rep_ratio']    = (df['seq_max_repetition'] / df['seq_length'].clip(lower=1)).astype('float32')
    df['seq_entropy_norm'] = (df['seq_entropy'] / np.log(df['seq_length'].clip(lower=2))).replace([np.inf,-np.inf],0).fillna(0).astype('float32')
    df['seq_first_is_key'] = df['seq_first'].astype(str).isin(key_all).astype('int8')
    df['seq_last_is_key']  = df['seq_last'].astype(str).isin(key_all).astype('int8')
    df['seq_first_eq_last']= (df['seq_first'].astype(str) == df['seq_last'].astype(str)).astype('int8')
    try:
        df['seq_len_bin']  = pd.qcut(df['seq_length'], q=5, labels=['len_vlow','len_low','len_mid','len_high','len_vhigh'], duplicates='drop').astype('category')
    except Exception:
        df['seq_len_bin']  = pd.cut(df['seq_length'], bins=[0,2,5,10,20,10**9], labels=['len_vlow','len_low','len_mid','len_high','len_vhigh']).astype('category')
    df['seq_has_any_key']  = (df['seq_total_key_counts_ext'] > 0).astype('int8')

    if drop_seq_after and 'seq' in df.columns:
        df.drop(columns=['seq'], inplace=True)

    # 시간/주기
    df['hour']        = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype('int16')
    df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype('int16')

    df['hour_sin'] = np.sin(2*np.pi * df['hour'].astype(np.float32) / 24.0).astype('float32')
    df['hour_cos'] = np.cos(2*np.pi * df['hour'].astype(np.float32) / 24.0).astype('float32')
    df['dow_sin']  = np.sin(2*np.pi * df['day_of_week'].astype(np.float32) / 7.0).astype('float32')
    df['dow_cos']  = np.cos(2*np.pi * df['day_of_week'].astype(np.float32) / 7.0).astype('float32')

    hour_of_week = (df['day_of_week'] - 1) * 24 + df['hour']
    df['week_hour_sin'] = np.sin(2*np.pi * hour_of_week / 168.0).astype('float32')
    df['week_hour_cos'] = np.cos(2*np.pi * hour_of_week / 168.0).astype('float32')

    hour_groups = {
        'deep_night': [0,1,2,3,4,5],
        'early_morning': [6,7,8],
        'morning': [9,10,11],
        'lunch': [12,13],
        'afternoon': [14,15,16,17],
        'evening': [18,19,20],
        'night': [21,22,23]
    }
    def get_hour_group(h):
        for g, arr in hour_groups.items():
            if h in arr: return g
        return 'unknown'
    df['hour_group'] = df['hour'].apply(get_hour_group).astype('category')

    df['is_weekend']   = df['day_of_week'].isin([6,7]).astype('int8')
    df['is_peak_hour'] = df['hour'].isin([8,9,12,13,18,19,20]).astype('int8')
    df['dow_hour_interaction'] = (df['day_of_week'].astype(str) + '_' + df['hour'].astype(str)).astype('category')

    # 카테고리 캐스트
    for c in ['inventory_id','age_group','gender']:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # 인벤토리 그룹
    high_perf_inv   = {'92','21','88'}
    good_perf_inv   = {'41','37','91'}
    medium_perf_inv = {'31','42','29'}
    if 'inventory_id' in df.columns:
        inv = df['inventory_id'].astype(str)
        grp = np.full(len(df), 'low_perf', dtype=object)
        grp[np.isin(inv, list(high_perf_inv))]   = 'high_perf'
        grp[np.isin(inv, list(good_perf_inv))]   = 'good_perf'
        grp[np.isin(inv, list(medium_perf_inv))] = 'medium_perf'
        df['inventory_group'] = pd.Series(grp, index=df.index).astype('category')

    bins = [-1, 6, 12, 18, 24]
    labels = ['night', 'morning', 'afternoon', 'evening']
    hour_bin = pd.cut(df['hour'], bins=bins, labels=labels).astype('category')

    if 'inventory_id' in df.columns:
        df['x_inv__hourbin'] = _cross_str(df['inventory_id'], hour_bin)
        if 'gender' in df.columns:
            df['x_inv__gender'] = _cross_str(df['inventory_id'], df['gender'])
        if 'age_group' in df.columns:
            df['x_inv__age'] = _cross_str(df['inventory_id'], df['age_group'])
        df['x_inv__dow'] = _cross_str(df['inventory_id'], df['day_of_week'].astype('category'))

    if 'age_group' in df.columns:
        df['x_age__hourbin'] = _cross_str(df['age_group'], hour_bin)
        df['x_age__dow'] = _cross_str(df['age_group'], df['day_of_week'].astype('category'))

    if 'inventory_group' in df.columns and 'age_group' in df.columns:
        df['x_invGroup__age'] = _cross_str(df['inventory_group'], df['age_group'])

    if 'seq_most_common' in df.columns and 'inventory_id' in df.columns:
        df['x_seqcommon__inv'] = _cross_str(df['seq_most_common'], df['inventory_id'])

    # 추가 교차
    if 'gender' in df.columns:
        df['x_gender__hourbin'] = _cross_str(df['gender'], hour_bin)
        df['x_gender__dow']     = _cross_str(df['gender'], df['day_of_week'].astype('category'))
    if 'age_group' in df.columns and 'gender' in df.columns:
        df['x_age__gender']     = _cross_str(df['age_group'], df['gender'])
    if 'inventory_group' in df.columns:
        df['x_invGroup__hourgroup'] = _cross_str(df['inventory_group'], df['hour_group'])
        df['x_invGroup__dow']       = _cross_str(df['inventory_group'], df['day_of_week'].astype('category'))
    df['x_hourgroup__dow'] = _cross_str(df['hour_group'], df['day_of_week'].astype('category'))

    # 수치 집계
    l_feat_cols = [c for c in df.columns if c.startswith('l_feat_')]
    if len(l_feat_cols) > 0:
        lf = df[l_feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['l_feat_sum']  = lf.sum(axis=1)
        df['l_feat_mean'] = lf.mean(axis=1)
        df['l_feat_max']  = lf.max(axis=1)
        df['l_feat_std']  = lf.std(axis=1).fillna(0.0)

    feat_a_cols = [c for c in df.columns if c.startswith('feat_a_')]
    if len(feat_a_cols) > 0:
        fa = df[feat_a_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['feat_a_sum']  = fa.sum(axis=1)
        df['feat_a_mean'] = fa.mean(axis=1)
        df['feat_a_max']  = fa.max(axis=1)
        df['feat_a_std']  = fa.std(axis=1).fillna(0.0)

    feat_b_cols = [c for c in df.columns if c.startswith('feat_b_')]
    if len(feat_b_cols) > 0:
        fb = df[feat_b_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['feat_b_sum']  = fb.sum(axis=1)
        df['feat_b_mean'] = fb.mean(axis=1)
        df['feat_b_max']  = fb.max(axis=1)

    feat_c_cols = [c for c in df.columns if c.startswith('feat_c_')]
    if len(feat_c_cols) > 0:
        fc = df[feat_c_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['feat_c_sum']  = fc.sum(axis=1)
        df['feat_c_mean'] = fc.mean(axis=1)
        df['feat_c_max']  = fc.max(axis=1)

    feat_d_cols = [c for c in df.columns if c.startswith('feat_d_')]
    if len(feat_d_cols) > 0:
        fd = df[feat_d_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['feat_d_sum']  = fd.sum(axis=1)
        df['feat_d_mean'] = fd.mean(axis=1)
        df['feat_d_max']  = fd.max(axis=1)

    feat_e_cols = [c for c in df.columns if c.startswith('feat_e_')]
    if len(feat_e_cols) > 0:
        fe = df[feat_e_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['feat_e_sum']  = fe.sum(axis=1)
        df['feat_e_mean'] = fe.mean(axis=1)
        df['feat_e_max']  = fe.max(axis=1)

    # HISTORY
    hist_cols = [c for c in df.columns if c.startswith('history_')]
    if len(hist_cols) > 0 and enable_history_aggs:
        h = df[hist_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        df['history_sum']  = h.sum(axis=1).astype('float32')
        df['history_mean'] = h.mean(axis=1).astype('float32')
        df['history_max']  = h.max(axis=1).astype('float32')
        df['history_std']  = h.std(axis=1).fillna(0.0).astype('float32')
        df['history_min']  = h.min(axis=1).astype('float32')
        df['history_q25']  = h.quantile(0.25, axis=1).astype('float32')
        df['history_q75']  = h.quantile(0.75, axis=1).astype('float32')

        a_cols = [c for c in hist_cols if c.startswith('history_a_')]
        b_cols = [c for c in hist_cols if c.startswith('history_b_')]

        if len(a_cols) > 0:
            ha = df[a_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
            df['hist_a_sum']  = ha.sum(axis=1)
            df['hist_a_mean'] = ha.mean(axis=1)
            df['hist_a_max']  = ha.max(axis=1)
            df['hist_a_std']  = ha.std(axis=1).fillna(0.0)
            df['hist_a_max_squared'] = (df['hist_a_max'] ** 2).astype('float32')
            df['hist_a_max_log1p']   = np.log1p(np.abs(df['hist_a_max']) + 1e-6).astype('float32')

        if len(b_cols) > 0:
            hb = df[b_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
            df['hist_b_sum']  = hb.sum(axis=1)
            df['hist_b_mean'] = hb.mean(axis=1)
            df['hist_b_max']  = hb.max(axis=1)
            df['hist_b_std']  = hb.std(axis=1).fillna(0.0)
            if len(b_cols) >= 5:
                recent = b_cols[-5:]
                df['hist_b_recent_mean'] = df[recent].mean(axis=1).astype('float32')

        df['hist_ab_sum']  = (df.get('hist_a_sum', 0) + df.get('hist_b_sum', 0)).astype('float32')
        df['hist_a_ratio'] = (df.get('hist_a_sum', 0) / (df['hist_ab_sum'] + 1e-6)).astype('float32')
        df['hist_b_ratio'] = (df.get('hist_b_sum', 0) / (df['hist_ab_sum'] + 1e-6)).astype('float32')
        df.drop(columns=hist_cols, inplace=True)

    # 수치 상호작용/로그 등
    def safe_ratio(a, b):
        return (a / (b + 1e-6)).astype('float32')

    for c in ['feat_e_1', 'feat_b_4', 'feat_d_2', 'feat_d_3',
              'feat_a_sum', 'feat_b_sum', 'feat_c_sum', 'feat_d_sum', 'feat_e_sum',
              'l_feat_sum', 'history_sum', 'history_max']:
        if c in df.columns:
            df[f'{c}_log1p'] = np.log1p(df[c].astype('float32').clip(lower=0))

    if {'l_feat_sum','history_mean'}.issubset(df.columns):
        df['lfeat_x_hist_mean'] = (df['l_feat_sum'] * df['history_mean']).astype('float32')
    if {'feat_e_sum','history_sum'}.issubset(df.columns):
        df['e_sum_x_hist_sum'] = (df['feat_e_sum'] * df['history_sum']).astype('float32')
    if {'feat_a_sum','l_feat_sum'}.issubset(df.columns):
        df['a_sum_div_l_sum'] = safe_ratio(df['feat_a_sum'], df['l_feat_sum'])
    if {'feat_b_sum','history_sum'}.issubset(df.columns):
        df['b_sum_div_hist_sum'] = safe_ratio(df['feat_b_sum'], df['history_sum'])
    if {'feat_c_sum','feat_d_sum'}.issubset(df.columns):
        df['c_sum_div_d_sum'] = safe_ratio(df['feat_c_sum'], df['feat_d_sum'])
    if {'feat_e_sum','feat_d_sum'}.issubset(df.columns):
        df['e_sum_div_d_sum'] = safe_ratio(df['feat_e_sum'], df['feat_d_sum'])
    if {'hist_b_recent_mean','hist_b_mean'}.issubset(df.columns):
        df['hist_b_recent_over_mean'] = safe_ratio(df['hist_b_recent_mean'], df['hist_b_mean'])
    if {'history_std','history_mean'}.issubset(df.columns):
        df['history_cv'] = safe_ratio(df['history_std'], df['history_mean'])
    if {'history_max','history_min'}.issubset(df.columns):
        df['history_range'] = (df['history_max'] - df['history_min']).astype('float32')
    if {'history_sum','feat_e_1'}.issubset(df.columns):
        df['hist_sum_div_e1'] = safe_ratio(df['history_sum'], df['feat_e_1'])
    if {'history_max','feat_d_2'}.issubset(df.columns):
        df['hist_max_x_d2'] = (df['history_max'] * df['feat_d_2']).astype('float32')

    return df


def safe_map_float(series: pd.Series, mapping: dict, fill_value: float) -> pd.Series:
    return series.astype('object').map(mapping).astype('float32').fillna(np.float32(fill_value))

def calculate_woe(df, cat_col, target_col, smooth=0.5):
    ct = pd.crosstab(df[cat_col], df[target_col])
    pos_count = ct.get(1, 0) + smooth
    neg_count = ct.get(0, 0) + smooth
    total_pos = df[target_col].sum() + smooth * len(ct)
    total_neg = (df[target_col] == 0).sum() + smooth * len(ct)
    woe = np.log((pos_count / total_pos) / (neg_count / total_neg))
    return woe.to_dict()

def apply_woe_encoding(df, woe_maps, suffix='__woe'):
    for col, mp in (woe_maps or {}).items():
        if col in df.columns:
            df[f'{col}{suffix}'] = safe_map_float(df[col], mp, 0.0)
    return df

