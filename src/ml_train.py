import datetime
import gc
import json
import warnings

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .ml_common import *

warnings.filterwarnings('ignore')

def _calc_metrics(y, p):
    if len(np.unique(y)) < 2:
        auc = np.nan; ap = np.nan
    else:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
    eps = 1e-15
    p_ = np.clip(p, eps, 1 - eps)
    pos = (y == 1); neg = ~pos
    pos_ll = -np.mean(np.log(p_[pos])) if pos.any() else 0.0
    neg_ll = -np.mean(np.log(1 - p_[neg])) if neg.any() else 0.0
    wll = 0.5 * pos_ll + 0.5 * neg_ll
    final = 0.5 * (ap if not pd.isna(ap) else 0.0) + 0.5 * (1 / (1 + wll))
    return auc, ap, wll, final

def save_group_reports(proc_all, y_all, oof_pred, out_dir, tag='bag1'):
    print("\n[REPORT] 그룹별 성능 요약 생성 중...")
    fields = [
        'inventory_id','inventory_group','age_group','gender','hour_group','day_of_week',
        'diversity_bin','seq_pattern_type','seq_first','seq_last','seq_most_common'
    ]
    overall_auc, overall_ap, overall_wll, overall_final = _calc_metrics(y_all, oof_pred)
    rows = []; min_cnt = 1000

    for col in fields:
        if col not in proc_all.columns: 
            continue
        g = proc_all[col].astype('object')
        grp = pd.DataFrame({'y': y_all, 'p': oof_pred, col: g})
        for v, sub in grp.groupby(col, sort=False):
            n = len(sub)
            if n < min_cnt: 
                continue
            auc, ap, wll, final = _calc_metrics(sub['y'].to_numpy(), sub['p'].to_numpy())
            rows.append({
                'field': col, 'value': str(v), 'count': n,
                'pos_rate': float(sub['y'].mean()),
                'auc': float(auc) if not pd.isna(auc) else np.nan,
                'ap': float(ap) if not pd.isna(ap) else np.nan,
                'wll': float(wll), 'final': float(final),
                'ap_delta_vs_overall': (float(ap) - overall_ap) if not pd.isna(ap) else np.nan,
                'final_delta_vs_overall': float(final) - overall_final
            })
    rpt = pd.DataFrame(rows).sort_values(['field','final_delta_vs_overall'])
    out1 = os.path.join(out_dir, f'cv_group_report__{tag}.csv')
    rpt.to_csv(out1, index=False)
    print(f"[REPORT] 저장: {out1}  (rows={len(rpt):,})")

    # 디사일 캘리브레이션
    dfc = pd.DataFrame({'y': y_all, 'p': oof_pred})
    dfc['decile'] = pd.qcut(dfc['p'].rank(method='first'), 10, labels=[f'd{i}' for i in range(1,11)])
    cal = dfc.groupby('decile').agg(
        n=('y','size'),
        mean_p=('p','mean'),
        pos_rate=('y','mean'),
        auc=('p', lambda x: np.nan)
    ).reset_index()
    out2 = os.path.join(out_dir, f'cv_calibration_deciles__{tag}.csv')
    cal.to_csv(out2, index=False)
    print(f"[REPORT] 저장: {out2}")

    # 터미널 요약
    if len(rpt) > 0:
        worst = rpt.sort_values('final_delta_vs_overall').head(10)[
            ['field','value','count','pos_rate','ap','final','final_delta_vs_overall']
        ]
        best  = rpt.sort_values('final_delta_vs_overall', ascending=False).head(10)[
            ['field','value','count','pos_rate','ap','final','final_delta_vs_overall']
        ]
        print("\n[REPORT] Worst 10 (final Δvs overall):")
        print(worst.to_string(index=False, justify='left', max_colwidth=20))
        print("\n[REPORT] Best 10 (final Δvs overall):")
        print(best.to_string(index=False, justify='left', max_colwidth=20))


def train_catboost_kfold(train_df, fold_name, enable_history_aggs, model_seed,
                            n_folds=5, fixed_kept_features=None, compute_fi_this_run=True,
                            bag_idx=1):
    print(f"\n=== {fold_name} 모델 학습(KFold={n_folds}, PRUNE=Bag1-Fold1 only, bottom-{N_PRUNE_BOTTOM}) ===")

    # 전처리(레이블 미사용 → 누수 없음)
    proc_all = create_revolutionary_features(train_df, enable_history_aggs=enable_history_aggs, drop_seq_after=True)
    base_cat_cols = [c for c in BASE_CAT_COLS if c in proc_all.columns]

    feat_cols_all = [c for c in proc_all.columns if c != 'clicked']
    X_all = proc_all[feat_cols_all].copy()
    y_all = proc_all['clicked'].astype('int8').to_numpy()

    # 디바이스 (torch 미설치여도 안전)
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    device_kwargs = {'task_type': 'GPU', 'devices': '0'} if has_gpu else {'task_type': 'CPU'}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=FOLD_SHUFFLE, random_state=FOLD_RANDOM_SEED)

    oof_pred = np.zeros(len(X_all), dtype=np.float32)
    kept_global = None
    if fixed_kept_features is not None:
        kept_global = list(fixed_kept_features)
        compute_fi_this_run = False

    for fi, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print("\n" + "-"*78)
        print(f"[{fold_name}] Fold {fi}/{n_folds}")

        X_tr = X_all.iloc[tr_idx].copy()
        X_val= X_all.iloc[va_idx].copy()
        y_tr = y_all[tr_idx]
        y_val= y_all[va_idx]

        # 인코딩 준비(폴드 train으로 적합)
        woe_fit_cols  = [c for c in WOE_TARGET_COLS  if c in X_tr.columns]
        te_fit_cols   = [c for c in TE_TARGET_COLS   if c in X_tr.columns]
        freq_fit_cols = [c for c in FREQ_TARGET_COLS if c in X_tr.columns]
        print(f"  인코딩: WOE:{len(woe_fit_cols)} TE:{len(te_fit_cols)} FREQ:{len(freq_fit_cols)}")

        woe_maps = {}
        for col in woe_fit_cols:
            train_with_target = pd.concat([X_tr[[col]], pd.Series(y_tr, name='clicked')], axis=1)
            woe_maps[col] = calculate_woe(train_with_target, col, 'clicked')
        X_tr = apply_woe_encoding(X_tr, woe_maps)
        X_val = apply_woe_encoding(X_val, woe_maps)

        te_maps = {}
        for col in te_fit_cols:
            train_with_target = pd.concat([X_tr[[col]], pd.Series(y_tr, name='clicked')], axis=1)
            mp, prior = calculate_target_encoding(train_with_target, col, 'clicked')
            te_maps[col] = {'map': mp, 'prior': prior}
        X_tr = apply_target_encoding(X_tr, te_maps)
        X_val = apply_target_encoding(X_val, te_maps)

        freq_maps = build_freq_maps(X_tr, freq_fit_cols)
        X_tr = apply_freq_maps(X_tr, freq_maps)
        X_val = apply_freq_maps(X_val, freq_maps)

        # 결측/카테고리
        cat_cols = [c for c in base_cat_cols if c in X_tr.columns]
        X_tr = sanitize_categoricals(X_tr, cat_cols)
        X_val = sanitize_categoricals(X_val, cat_cols)
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        for c in num_cols:
            if X_tr[c].dtype.kind in 'biufc':
                med = X_tr[c].median()
                X_tr[c] = X_tr[c].fillna(med)
                X_val[c] = X_val[c].fillna(med)

        # Full-fit은 Bag1-Fold1에서만(kept 산출용)
        do_full = (kept_global is None) and compute_fi_this_run
        if do_full:
            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool   = Pool(X_val, y_val, cat_features=cat_cols)
            model = CatBoostClassifier(**CB_PARAMS, **device_kwargs, random_seed=model_seed + fi)
            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                plot=False,
                early_stopping_rounds=300,
                metric_period=50,
                verbose=100
            )
            y_hat_full = model.predict_proba(val_pool)[:,1]
            print(f"  [Fold {fi} Full] AUC:{roc_auc_score(y_val, y_hat_full):.6f}")

            # 중요도 산출 → 하위 10개 제거
            imp = model.get_feature_importance(type="PredictionValuesChange", data=val_pool)
            fi_series = pd.Series(np.abs(imp), index=X_tr.columns).sort_values(ascending=False)
            n_drop = min(N_PRUNE_BOTTOM, len(fi_series))
            drop_feats = fi_series.tail(n_drop).index.tolist()
            kept_global = [f for f in fi_series.index.tolist() if f not in drop_feats]
            print(f"  [PRUNE] (Bag1-Fold1) 하위 {n_drop}개 드롭 → 보존 {len(kept_global)}개")
        else:
            print("  [Full-fit] 생략 → kept로 Refit만 수행")

        # 공통 kept로 Refit
        kept = [c for c in kept_global if c in X_tr.columns]
        X_tr2  = X_tr[kept].copy()
        X_val2 = X_val[kept].copy()
        cat_cols2 = [c for c in cat_cols if c in kept]
        train_pool2 = Pool(X_tr2, y_tr, cat_features=cat_cols2)
        val_pool2   = Pool(X_val2, y_val, cat_features=cat_cols2)

        model2 = CatBoostClassifier(**CB_PARAMS, **device_kwargs, random_seed=model_seed + 10_000 + fi)
        model2.fit(
            train_pool2,
            eval_set=val_pool2,
            use_best_model=True,
            plot=False,
            early_stopping_rounds=300,
            metric_period=50,
            verbose=100
        )
        y_hat2 = model2.predict_proba(val_pool2)[:,1]
        print(f"  [Fold {fi} ReFit] AUC:{roc_auc_score(y_val, y_hat2):.6f}  | kept:{len(kept)}")

        # (저장) 모델/인코딩맵
        model_path = os.path.join(CB_DIR, f"b{bag_idx:02d}_f{fi:02d}.cbm")
        model2.save_model(model_path)
        enc_maps = {'woe': woe_maps, 'te': te_maps, 'freq': freq_maps}
        with open(os.path.join(ENC_DIR, f"b{bag_idx:02d}_f{fi:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(enc_maps, f, ensure_ascii=False)

        # OOF 저장
        oof_pred[va_idx] = y_hat2

        del X_tr, X_val, X_tr2, X_val2, train_pool2, val_pool2, model2
        if do_full:
            del train_pool, val_pool, model
        gc.collect()

    # OOF/리포트
    oof_scores = calculate_competition_score(y_all, oof_pred)
    oof_auc = roc_auc_score(y_all, oof_pred)
    print("\n" + "-"*78)
    print(f"[{fold_name}] OOF Final:{oof_scores['final_score']:.6f} | AP:{oof_scores['ap']:.6f} | WLL:{oof_scores['wll']:.6f} | AUC:{oof_auc:.6f}")

    tag = fold_name.replace(' ','_')
    save_group_reports(proc_all, y_all, oof_pred, REPORT_DIR, tag=tag)

    return {
        'oof_pred': oof_pred,
        'oof_scores': oof_scores,
        'kept_features': kept_global
    }



def run_training():
    ensure_directories()
    set_global_seed(GLOBAL_SEED)
    print('=' * 80)
    print('Enhanced 10-bag V5 (Train-Only) - Bag1-Fold1 PRUNE bottom-10 + KFold')
    print('=' * 80)

    train_raw_cols = get_raw_columns(TRAIN_PATH)
    enable_history = any(c.startswith('history_') for c in train_raw_cols)
    print(f'[feature-compat] ENABLE_HISTORY_AGGS={enable_history}')

    total_pos, total_neg = count_pos_neg(TRAIN_PATH)
    pi_global = total_pos / max(1, (total_pos + total_neg))
    per_bag_neg = int(total_pos * NEG_RATIO)
    if per_bag_neg <= 0:
        raise ValueError('per_bag_neg가 0입니다.')
    print(f'[counts] total_pos={total_pos:,}  total_neg={total_neg:,}  π_global={pi_global:.6f}')
    print(f'[plan] N_BAGS={N_BAGS}  per_bag_neg={per_bag_neg:,}  기대 음성 사용≈{per_bag_neg * N_BAGS:,}')

    pos_path, neg_paths = materialize_pos_and_neg_bags(TRAIN_PATH, USECOLS, total_pos, total_neg, per_bag_neg, N_BAGS, RANDOM_SEED, STAGING_DIR)
    actual_bags = len(neg_paths)
    if actual_bags < N_BAGS:
        print(f'[경고] 실제 생성된 bag 개수={actual_bags} (요청={N_BAGS})')

    pos_df = pd.read_parquet(pos_path)
    print(f'[pos] loaded: {len(pos_df):,}')
    kept_from_bag1 = None

    for bag in range(actual_bags):
        print('\n' + '=' * 80)
        print(f'========  BAG {bag+1}/{actual_bags}  (Train-Only)  ========')
        print('=' * 80)

        neg_df = pd.read_parquet(neg_paths[bag])
        print(f'[neg bag {bag+1}] loaded: {len(neg_df):,}')
        fold_df = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f'[bag {bag+1}] rows={len(fold_df):,}  pos={len(pos_df):,}  neg={len(neg_df):,}')

        if bag == 0:
            artifacts = train_catboost_kfold(fold_df, f'Bag{bag+1}_V5', enable_history, RANDOM_SEED + bag, n_folds=N_FOLDS, fixed_kept_features=None, compute_fi_this_run=True, bag_idx=bag+1)
            kept_from_bag1 = artifacts['kept_features']
            if kept_from_bag1:
                with open(KEPT_PATH, 'w', encoding='utf-8') as f:
                    for c in kept_from_bag1:
                        f.write(c + '\n')
        else:
            artifacts = train_catboost_kfold(fold_df, f'Bag{bag+1}_V5', enable_history, RANDOM_SEED + bag, n_folds=N_FOLDS, fixed_kept_features=kept_from_bag1, compute_fi_this_run=False, bag_idx=bag+1)

        del neg_df, fold_df, artifacts
        gc.collect()

    meta = {
        'mode': 'train_only',
        'n_bags': actual_bags,
        'n_folds': N_FOLDS,
        'neg_ratio': NEG_RATIO,
        'random_seed': RANDOM_SEED,
        'fold_random_seed': FOLD_RANDOM_SEED,
        'cb_params': CB_PARAMS,
        'equal_avg_models': actual_bags * N_FOLDS,
        'feature_version': 'v5',
        'saved_at': datetime.datetime.now().isoformat(),
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 80)
    print(f'Train-Only 완료 - {actual_bags} bags × {N_FOLDS} folds → 모델/인코딩맵 저장됨')
    print('=' * 80)
