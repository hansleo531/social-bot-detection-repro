# -*- coding: utf-8 -*-
"""域 0 / 域 6 画像与典型案例探索"""
import numpy as np
import pandas as pd

BASE = r"D:\botrans\data\{d}\raw\group{d}_raw_feat.csv"
NODES = r"D:\botrans\data\{d}\raw\group{d}_nodes.csv"
EDGES = r"D:\botrans\data\{d}\raw\group{d}_edges_new.csv"

# Twitter snowflake -> 注册时间
def snowflake_time(uid):
    try:
        n = int(str(uid).lstrip("u"))
        ms = (n >> 22) + 1288834974657
        return pd.to_datetime(ms, unit="ms", errors="coerce")
    except Exception:
        return pd.NaT


for d in (0, 6):
    feat = pd.read_csv(BASE.format(d=d))
    nodes = pd.read_csv(NODES.format(d=d))
    edges = pd.read_csv(EDGES.format(d=d))
    df = feat.merge(nodes, on="id", how="inner")
    df["created"] = df["id"].map(snowflake_time)
    df["year"] = df["created"].dt.year

    print(f"\n===== domain {d} =====")
    print("users", len(df), "bots", int((df.label == 1).sum()),
          f"({(df.label==1).mean():.1%})")
    print("edges", len(edges), "avg_deg", round(len(edges) * 2 / len(df), 2))
    for c in ["followers_count", "following_count", "tweet_count", "active_days",
              "listed_count", "name_len"]:
        q = df[c].quantile([0.25, 0.5, 0.75]).round(1).tolist()
        print(f"  {c:16s} median={df[c].median():>10.1f}  q25/q75={q}")
    print("  verified", f"{df.verified.mean():.1%}", " default_pic", f"{df.picture.mean():.1%}",
          " protected", f"{df.protected.mean():.1%}")
    print("  year dist:", df["year"].value_counts().sort_index().tail(8).to_dict())

    bot = df[df.label == 1].copy()
    bot["ff_ratio"] = bot.following_count / (bot.followers_count + 1)
    print("  bots with 0 followers:", int((bot.followers_count == 0).sum()),
          " verified bots:", int(bot.verified.sum()))
    print("  top tweet_count bots:")
    for _, r in bot.nlargest(3, "tweet_count").iterrows():
        print(f"    {r.id} tweets={int(r.tweet_count)} followers={int(r.followers_count)} "
              f"following={int(r.following_count)} days={int(r.active_days)} year={r.year}")
    print("  top ff_ratio bots:")
    for _, r in bot.nlargest(3, "ff_ratio").iterrows():
        print(f"    {r.id} following={int(r.following_count)} followers={int(r.followers_count)} "
              f"tweets={int(r.tweet_count)} days={int(r.active_days)} year={r.year}")
