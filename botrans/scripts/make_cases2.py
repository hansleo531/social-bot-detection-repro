# -*- coding: utf-8 -*-
"""域 0 / 域 6 案例筛选（修正版）"""
import pandas as pd

FEAT = ["followers_count", "following_count", "active_days", "name_len",
        "uname_len", "tweet_count", "listed_count", "protected", "verified", "picture"]


def load(d):
    feat = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_raw_feat.csv")
    nodes = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_nodes.csv")
    edges = pd.read_csv(rf"D:\botrans\data\{d}\raw\group{d}_edges_new.csv")
    df = feat.merge(nodes, on="id", how="inner")
    deg = pd.concat([edges["source"], edges["target"]]).value_counts()
    df["degree"] = df["id"].map(deg).fillna(0).astype(int)
    return df


def one(df, cond, col="followers_count", ascending=False, tag="", exclude=None):
    sub = df[cond]
    if exclude is not None:
        sub = sub[~sub.id.isin(exclude)]
    if len(sub) == 0:
        return None
    r = sub.sort_values(col, ascending=ascending).iloc[[0]].copy()
    r["类型"] = tag
    return r


def cases_for(d):
    df = load(d)
    bot = df.label == 1
    human = df.label == 0
    med = df[human].followers_count.median()
    idx = (df[human].followers_count - med).abs().idxmin()

    used = set()
    picks = []
    for spec in [
        (bot, "followers_count", False, "粉丝最多的机器人"),
        (bot, "tweet_count", False, "发帖量最大的机器人"),
        (bot & (df.followers_count == 0), "following_count", False, "零粉丝高关注（钓鱼型）"),
        (bot & (df.verified == 1), "followers_count", False, "认证账号中的机器人"),
    ]:
        r = one(df, spec[0], spec[1], spec[2], spec[3], exclude=used)
        if r is not None:
            used.add(r.iloc[0]["id"])
            picks.append(r)
    picks.append(df.loc[[idx]].assign(类型="对照：粉丝数中位的人类"))
    out = pd.concat([p for p in picks if p is not None])
    out = out.drop_duplicates(subset=["类型", "id"])
    return out, df


for d in (0, 6):
    out, df = cases_for(d)
    out.to_csv(rf"D:\botrans\cases_{d}.csv", index=False, encoding="utf-8")
    print(f"=== domain {d} ===")
    print(out[["类型", "id", "followers_count", "following_count", "tweet_count",
               "active_days", "listed_count", "verified", "degree", "label"]].to_string(index=False))

