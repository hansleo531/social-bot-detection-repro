# -*- coding: utf-8 -*-
"""域 0 / 域 6：特征可分性 + 典型案例筛选"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEAT = ["followers_count", "following_count", "active_days", "name_len",
        "uname_len", "tweet_count", "listed_count", "protected", "verified", "picture"]


def load(d):
    feat = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_raw_feat.csv")
    nodes = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_nodes.csv")
    return feat.merge(nodes, on="id", how="inner")


for d in (0, 6):
    df = load(d)
    X = np.log1p(df[FEAT].astype(float).clip(lower=0))
    y = df["label"].values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
    print(f"domain {d}: 仅用元数据的 LR 5 折 AUC = {auc:.3f}")

    # 人机特征均值对比
    print("   human vs bot means:")
    for c in ["followers_count", "tweet_count", "active_days", "listed_count", "verified"]:
        h = df[df.label == 0][c].median()
        b = df[df.label == 1][c].median()
        print(f"     {c:16s} human={h:>10.1f}  bot={b:>10.1f}")
