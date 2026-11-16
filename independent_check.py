"""Independent re-implementation of the DSS calculations in plain Python (no NumPy, loops only).
It is used to verify dss_core.py: both must give the same weights, closeness values and SAW scores."""
import math


def _gm(row):
    p = 1.0
    for x in row:
        p *= x
    return p ** (1.0 / len(row))


def fuzzy_weights(M, delta):
    n = len(M)
    if n == 1:
        return [1.0]
    Lm = [[1.0] * n for _ in range(n)]
    Um = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = M[i][j]
            if a >= 1:
                lo, up = max(1.0, a - delta), a + delta
            else:
                lo, up = 1.0 / (1.0 / a + delta), 1.0 / max(1.0, 1.0 / a - delta)
            Lm[i][j], Um[i][j] = lo, up
            Lm[j][i], Um[j][i] = 1.0 / up, 1.0 / lo
    rl = [_gm(r) for r in Lm]
    rm = [_gm(r) for r in M]
    ru = [_gm(r) for r in Um]
    sl, sm, su = sum(rl), sum(rm), sum(ru)
    d = [(rl[i] / su + rm[i] / sm + ru[i] / sl) / 3.0 for i in range(n)]
    return [x / sum(d) for x in d]


def matrix(n, judg):
    M = [[1.0] * n for _ in range(n)]
    for p, v in judg.items():
        i, j = (int(t) for t in p.split("-"))
        M[i][j], M[j][i] = float(v), 1.0 / float(v)
    return M


def topsis(X, w, benefit):
    na, nc = len(X), len(X[0])
    norm = [math.sqrt(sum(X[i][j] ** 2 for i in range(na))) for j in range(nc)]
    V = [[X[i][j] / norm[j] * w[j] for j in range(nc)] for i in range(na)]
    pis, nis = [], []
    for j in range(nc):
        col = [V[i][j] for i in range(na)]
        pis.append(max(col) if benefit[j] else min(col))
        nis.append(min(col) if benefit[j] else max(col))
    out = []
    for i in range(na):
        dp = math.sqrt(sum((V[i][j] - pis[j]) ** 2 for j in range(nc)))
        dn = math.sqrt(sum((V[i][j] - nis[j]) ** 2 for j in range(nc)))
        out.append(dn / (dp + dn))
    return out


def saw(X, w, benefit):
    na, nc = len(X), len(X[0])
    out = []
    for i in range(na):
        s = 0.0
        for j in range(nc):
            col = [X[k][j] for k in range(na)]
            s += w[j] * (X[i][j] / max(col) if benefit[j] else min(col) / X[i][j])
        out.append(s)
    return out


def full_run(vals, J, delta, groups_of, benefit, sizes, scen_group_weights):
    """vals[criterion][alternative]; returns baseline weights, baseline closeness, SAW and scenario closeness."""
    X = [[vals[j][i] for j in range(len(vals))] for i in range(len(vals[0]))]
    g = fuzzy_weights(matrix(4, J["top"]), delta)
    loc = fuzzy_weights(matrix(3, J["dem"]), delta) + [1.0] + fuzzy_weights(matrix(2, J["com"]), delta) + fuzzy_weights(matrix(2, J["acc"]), delta)
    wfun = lambda gs: [gs[groups_of[j]] * loc[j] for j in range(len(loc))]
    base = wfun(g)
    scen = {k: topsis(X, wfun(gk(g) if callable(gk) else gk), benefit) for k, gk in scen_group_weights.items()}
    return {"w": base, "C": topsis(X, base, benefit), "saw": saw(X, base, benefit), "scen": scen}
