# -*- coding: utf-8 -*-
"""生成带派生指标的案例表（供 Word 使用）"""
import numpy as np
import pandas as pd

EPOCH = 1288834974657


def snowflake_dt(uid):
    try:
        n = int(str(uid).lstrip("u"))
        if n < 2 ** 22:
            return pd.NaT
        return pd.to_datetime((n >> 22) + EPOCH, unit="ms", errors="coerce")
    except Exception:
        return pd.NaT


def build(d, csv):
    feat = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_raw_feat.csv")
    nodes = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_nodes.csv")
    edges = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_edges_new.csv")
    df = feat.merge(nodes, on="id", how="inner").set_index("id")

    out_deg = edges["source"].value_counts()
    in_deg = edges["target"].value_counts()
    df["out_deg"] = df.index.map(out_deg).fillna(0).astype(int)
    df["in_deg"] = df.index.map(in_deg).fillna(0).astype(int)

    label = df["label"].to_dict()
    nbr = {}
    for s, t in edges[["source", "target"]].itertuples(index=False):
        nbr.setdefault(s, set()).add(t)
        nbr.setdefault(t, set()).add(s)
    ratio, ncnt = {}, {}
    for nid in df.index:
        ns = [x for x in nbr.get(nid, set()) if x in label]
        ncnt[nid] = len(ns)
        ratio[nid] = np.mean([label[x] for x in ns]) if ns else np.nan
    df["nbr_bot_ratio"] = pd.Series(ratio)
    df["nbr_count"] = pd.Series(ncnt)

    df["created"] = [snowflake_dt(i) for i in df.index]
    df["age_days"] = (pd.Timestamp("2023-12-31") - df["created"]).dt.days
    df["ff_ratio"] = df["following_count"] / (df["followers_count"] + 1)
    df["tweets_per_day"] = df["tweet_count"] / df["active_days"].clip(lower=1)
    df["age_text"] = [
        c.strftime("%Y-%m") if pd.notna(c) else "—" for c in df["created"]
    ]
    df["followers_pct"] = df["followers_count"].rank(pct=True) * 100
    df["tweets_pct"] = df["tweet_count"].rank(pct=True) * 100

    cases = pd.read_csv(csv)
    rows = []
    for _, r in cases.iterrows():
        a = df.loc[r["id"]]
        rows.append({
            "类型": r["类型"],
            "id": r["id"],
            "followers": int(a.followers_count),
            "following": int(a.following_count),
            "tweets": int(a.tweet_count),
            "days": int(a.active_days),
            "listed": int(a.listed_count),
            "verified": int(a.verified),
            "label": int(a.label),
            "created": a.age_text,
            "tpd": round(float(a.tweets_per_day), 1),
            "ff": round(float(a.ff_ratio), 2),
            "in_deg": int(a.in_deg),
            "out_deg": int(a.out_deg),
            "nbr_bot": round(float(a.nbr_bot_ratio), 2) if pd.notna(a.nbr_bot_ratio) else -1,
            "f_pct": round(float(a.followers_pct)),
            "t_pct": round(float(a.tweets_pct)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(rf"D:\botrans\cases_enriched_{d}.csv", index=False, encoding="utf-8")
    print(out.to_string(index=False))
    return out


build(0, r"D:\botrans\cases_0.csv")
build(6, r"D:\botrans\cases_6.csv")
