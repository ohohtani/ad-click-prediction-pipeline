# blend_logit_w75_25.py
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd

# ====== 설정 ======
FILE1   = "outputs/final_submission.csv"      # 첫 번째 제출/예측 CSV (catboost)
FILE2   = "outputs/wide_deep_inference.csv"   # 두 번째 제출/예측 CSV (wide & deep)
OUTDIR  = "./outputs"
OUT_SUB = os.path.join(OUTDIR, "submission_logit_w75_25.csv")

# 자동 탐지 후보
ID_CANDS   = ["ID","id","Id","sample_id","row_id","index","Index"]
PROB_CANDS = ["clicked","prob","probability","prediction","pred","score","y_pred","yhat","target","value"]

# ====== 유틸 ======
def read_csv_auto(path):
    """인코딩 자동 시도"""
    for enc in ["utf-8","cp949","euc-kr","latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def pick_two():
    """FILE1/FILE2가 존재하면 사용, 아니면 최신 CSV 2개 자동 선택"""
    if os.path.exists(FILE1) and os.path.exists(FILE2):
        return FILE1, FILE2
    csvs = [p for p in glob.glob("*.csv") if os.path.isfile(p)]
    if len(csvs) < 2:
        raise FileNotFoundError("CSV 최소 2개 필요")
    csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return csvs[0], csvs[1]

def find_col(cols, cands):
    low = [c.lower() for c in cols]
    for cand in cands:
        if cand.lower() in low:
            return cols[low.index(cand.lower())]
    return None

def autodetect(df):
    """ID/확률 컬럼 자동 탐지 (확률이 없으면 [0,1] 비율 가장 높은 수치열 추정)"""
    id_col = find_col(df.columns, ID_CANDS)
    prob_col = find_col(df.columns, PROB_CANDS)
    if prob_col is None:
        best, br = None, -1.0
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0: 
                continue
            if s.nunique() <= 2:  # 이진은 제외
                continue
            r = ((s >= 0) & (s <= 1)).mean()
            if r > br:
                br = r; best = c
        prob_col = best
    return id_col, prob_col

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def weighted_logit_average(p1, p2, w1=0.75, w2=0.25):
    """로짓(오즈) 공간 가중 평균 후 시그모이드 역변환"""
    z1 = logit(p1); z2 = logit(p2)
    z  = (w1 * z1 + w2 * z2) / (w1 + w2 + 1e-12)
    return sigmoid(z)

# ====== 메인 ======
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    f1, f2 = pick_two()
    print(f"[입력] {f1} vs {f2}")

    df1 = read_csv_auto(f1); df2 = read_csv_auto(f2)
    id1, pr1 = autodetect(df1); id2, pr2 = autodetect(df2)
    if pr1 is None or pr2 is None:
        raise ValueError("확률 컬럼 자동탐지 실패 (PROB_CANDS를 보강하세요).")

    # ID가 없으면 위치 기준으로 맞춰서 병합
    if id1 is None or id2 is None:
        n = min(len(df1), len(df2))
        ids = np.arange(n)
        p1  = pd.to_numeric(df1[pr1], errors="coerce").values[:n]
        p2  = pd.to_numeric(df2[pr2], errors="coerce").values[:n]
    else:
        a = pd.DataFrame({"ID": df1[id1], "p1": pd.to_numeric(df1[pr1], errors="coerce")})
        b = pd.DataFrame({"ID": df2[id2], "p2": pd.to_numeric(df2[pr2], errors="coerce")})
        m = a.merge(b, on="ID", how="inner")
        ids = m["ID"].values
        p1  = m["p1"].clip(0, 1).values
        p2  = m["p2"].clip(0, 1).values

    print(f"[컬럼] file1: id='{id1}' prob='{pr1}',  file2: id='{id2}' prob='{pr2}'")
    print(f"[병합] 공통 샘플 수: {len(ids):,}")

    probs = weighted_logit_average(p1, p2, 0.75, 0.25)
    out_df = pd.DataFrame({"ID": ids, "clicked": np.clip(probs, 1e-6, 1 - 1e-6)})
    out_df.to_csv(OUT_SUB, index=False)
    print(f"[저장] {OUT_SUB}")
    print("완료.")

if __name__ == "__main__":
    main()
