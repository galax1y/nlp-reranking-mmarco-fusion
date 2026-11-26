import sys, re
import numpy as np, pandas as pd
R = sys.argv[1]
df = pd.read_csv(f"{R}/ranks_por_pergunta_com_cascata.csv", dtype={"qid": str})
n = len(df)
cols = [c for c in df.columns if c.startswith("rank_")]
def mrr(r): r = np.asarray(r); return np.where((r > 0) & (r <= 10), 1.0 / np.where(r > 0, r, 1), 0.0)
M = {c[5:]: mrr(df[c]) for c in cols}

# reproduce SUB exactly as the notebook did
SUB = np.sort(np.random.default_rng(42).choice(n, size=1000, replace=False))
print("== check: SUB means vs notebook (BM25 0.1423, CE nosso K100 0.2550, direct n/a)")
print("BM25 SUB", M["BM25"][SUB].mean().round(4), "| CE nosso SUB", M["CE nosso (K=100)"][SUB].mean().round(4),
      "| CE pronto SUB", M["CE pronto (K=100)"][SUB].mean().round(4))
print("full-set check BM25", M["BM25"].mean().round(4), "CE pronto", M["CE pronto (K=100)"].mean().round(4))

# ---- ceiling / oracle
b = df["rank_BM25"].values
in100 = (b > 0) & (b <= 100)   # first relevant in BM25 top-100 (note: rank_BM25 is in top-100 list only)
print("\n== ceiling")
print("queries with >=1 relevant in BM25 top-100:", in100.mean().round(4), f"({in100.sum()} / {n})")
oracle = in100.mean()   # perfect reranker MRR@10 (relevant put at rank 1 when reachable)
for s in ["BM25", "CE nosso (K=100)", "CE pronto (K=100)", "CE nosso (K=100) → monoPTT5 (N=20)"]:
    m = M[s].mean()
    print(f"{s:40s} MRR {m:.4f} | % of oracle {m/oracle:6.1%} | gap closed {(m-M['BM25'].mean())/(oracle-M['BM25'].mean()):6.1%}"
          f" | MRR on reachable q {M[s][in100].mean():.4f} | rank1 on reachable {(df['rank_'+s].values[in100]==1).mean():.3f}")

# ---- where the relevant sat in BM25 and what the CE did (rescue depth)
print("\n== rescues by BM25 depth of first relevant (CE pronto K=100 / CE nosso K=100): share ending in top-10")
for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 25), (26, 50), (51, 100)]:
    m = (b >= lo) & (b <= hi)
    p, o = df["rank_CE pronto (K=100)"].values[m], df["rank_CE nosso (K=100)"].values[m]
    top10 = lambda r: ((r > 0) & (r <= 10)).mean()
    print(f"BM25 rank {lo:>3}-{hi:<3} n={m.sum():5d} | pronto top10 {top10(p):.3f} rank1 {(p==1).mean():.3f} | nosso top10 {top10(o):.3f} rank1 {(o==1).mean():.3f}")

# ---- query types
q = df["query"].str.lower().str.strip()
def tipo(s):
    for k, pat in [("o que / o que é", r"^(o )?que (é|e|são|significa)\b|^o que\b"), ("como", r"^como\b"),
                   ("qual / quais", r"^qua(l|is)\b"), ("quanto(s/as)", r"^quant[oa]s?\b"), ("quando", r"^quando\b"),
                   ("onde", r"^onde\b|^de onde\b"), ("quem", r"^quem\b"), ("por que", r"^por ?qu[eê]\b"),
                   ("definição / significado", r"defini|signific")]:
        if re.search(pat, s): return k
    return "outras (palavras-chave etc.)"
df["tipo"] = q.map(tipo)
df["n_palavras"] = q.str.split().str.len()
df["tem_numero"] = q.str.contains(r"\d")
print("\n== by query type (MRR@10)  [n]  BM25 | pronto | nosso | nosso→T5(20) | gain pronto-BM25")
for t, g in df.groupby("tipo"):
    i = g.index.values
    print(f"{t:30s} n={len(i):5d} | {M['BM25'][i].mean():.3f} | {M['CE pronto (K=100)'][i].mean():.3f} | "
          f"{M['CE nosso (K=100)'][i].mean():.3f} | {M['CE nosso (K=100) → monoPTT5 (N=20)'][i].mean():.3f} | "
          f"+{M['CE pronto (K=100)'][i].mean()-M['BM25'][i].mean():.3f} | reachable {in100[i].mean():.2f}")
print("\n== by query length")
for lab, m in [("1-3 palavras", df.n_palavras <= 3), ("4-6", df.n_palavras.between(4, 6)), ("7+", df.n_palavras >= 7)]:
    i = df.index[m].values
    print(f"{lab:14s} n={len(i):5d} | BM25 {M['BM25'][i].mean():.3f} | pronto {M['CE pronto (K=100)'][i].mean():.3f} | "
          f"nosso {M['CE nosso (K=100)'][i].mean():.3f} | gain pronto {M['CE pronto (K=100)'][i].mean()-M['BM25'][i].mean():+.3f} | reachable {in100[i].mean():.2f}")
for lab, m in [("com número", df.tem_numero), ("sem número", ~df.tem_numero)]:
    i = df.index[m].values
    print(f"{lab:14s} n={len(i):5d} | BM25 {M['BM25'][i].mean():.3f} | pronto {M['CE pronto (K=100)'][i].mean():.3f} | "
          f"nosso {M['CE nosso (K=100)'][i].mean():.3f} | gain {M['CE pronto (K=100)'][i].mean()-M['BM25'][i].mean():+.3f}")

# ---- cascade wins/losses
print("\n== cascade (3rd stage) vs its CE")
inf = lambda r: np.where(np.asarray(r) > 0, r, 10**6)
for ce in ["CE nosso", "CE pronto"]:
    for N in [5, 10, 20]:
        a, c = inf(df[f"rank_{ce} (K=100) → monoPTT5 (N={N})"]), inf(df[f"rank_{ce} (K=100)"])
        print(f"{ce} N={N:2d}: wins {(a<c).sum():4d} ties {(a==c).sum():4d} losses {(a>c).sum():4d} | "
              f"rank1 gained {((a==1)&(c!=1)).sum():4d} lost {((a!=1)&(c==1)).sum():4d}")

# ---- agreement between the two CEs
a, c = inf(df["rank_CE nosso (K=100)"]), inf(df["rank_CE pronto (K=100)"])
both = (a == 1) & (c == 1)
print(f"\n== rank-1 overlap: both {both.sum()} | only pronto {((c==1)&(a!=1)).sum()} | only nosso {((a==1)&(c!=1)).sum()}")

# ---- K curve: share of ceiling converted
print("\n== K sweep: Recall@10 after rerank / BM25 Recall@K ceiling")
ceil = {10: .3105, 25: .4109, 50: .4836, 100: .5566}
tab = pd.read_csv(f"{R}/tabela_resultados.csv")
for _, r in tab.iterrows():
    k = re.search(r"K=(\d+)", r.Sistema)
    if k: print(f"{r.Sistema:22s} Recall@10 {r['Recall@10']:.4f} / ceiling {ceil[int(k.group(1))]:.4f} = {r['Recall@10']/ceil[int(k.group(1))]:.1%}")

# ---- examples: queries where cascade on CE nosso rescued rank-1, and losses
ex = df[(inf(df["rank_CE nosso (K=100) → monoPTT5 (N=20)"]) == 1) & (inf(df["rank_CE nosso (K=100)"]) > 5)].head(5)
print("\n== examples cascade wins (nosso rank>5 -> 1):")
print(ex[["qid", "query", "rank_BM25", "rank_CE nosso (K=100)", "rank_CE pronto (K=100)"]].to_string(index=False))
