"""Retail Location DSS - calculation core (Fuzzy AHP + TOPSIS, SAW cross-check).
All case data are HYPOTHETICAL. Only NumPy is required."""
import numpy as np
import independent_check as ind
from case_text import DEFN, LOC_TEXT, REASONS

ALT = ["City Centre", "Recreation & Tourism", "Transit Hub", "Residential"]
GROUPS = ["Demand", "Cost", "Competition", "Accessibility"]
# id, group index, name, unit, type, values (CC, R&T, TH, Res), min, max
CRIT = [
    ("C1.1", 0, "Foot Traffic", "people/day", "Benefit", [14000, 9000, 18000, 3500], 0, None),
    ("C1.2", 0, "Population", "people within 1 km", "Benefit", [22000, 9000, 15000, 18000], 0, None),
    ("C1.3", 0, "Buying Power", "index 0-10", "Benefit", [8, 7, 6, 6], 0, 10),
    ("C2.1", 1, "Rent", "EUR/m2/year", "Cost", [450, 320, 380, 150], 0, None),
    ("C3.1", 2, "Competitors", "count within 1 km", "Cost", [9, 5, 7, 2], 0, None),
    ("C3.2", 2, "Anchor Stores", "count within 1 km", "Benefit", [6, 4, 3, 2], 0, None),
    ("C4.1", 3, "Transit Distance", "metres", "Cost", [200, 500, 50, 400], 1, None),
    ("C4.2", 3, "Road Access", "scale 1-9", "Benefit", [3, 6, 4, 8], 1, 9),
]
NC, NA = len(CRIT), len(ALT)
GO = [c[1] for c in CRIT]
BEN = np.array([c[4] == "Benefit" for c in CRIT])
DEFAULT_VALS = [list(map(float, c[5])) for c in CRIT]          # [criterion][alternative]
PAIRS = {  # key: (item names, default upper-triangle judgements)
    "top": (GROUPS, {"0-1": 2, "0-2": 3, "0-3": 3, "1-2": 2, "1-3": 2, "2-3": 1}),
    "dem": (["Foot Traffic", "Population", "Buying Power"], {"0-1": 2, "0-2": 3, "1-2": 2}),
    "com": (["Competitors", "Anchor Stores"], {"0-1": 2}),
    "acc": (["Transit Distance", "Road Access"], {"0-1": 2}),
}
DEFAULT_J = {k: dict(v[1]) for k, v in PAIRS.items()}
SCEN = [  # key, name, preset group weights (None = from AHP / custom)
    ("S1", "S1 Baseline (Fuzzy AHP)", None),
    ("S2", "S2 Cost-focused", [.20, .50, .15, .15]),
    ("S3", "S3 Demand-focused", [.60, .15, .10, .15]),
    ("S4", "S4 Access-focused", [.25, .15, .10, .50]),
    ("S5", "S5 Competition-focused (extra)", [.20, .15, .50, .15]),
    ("C", "Custom group weights", None),
]
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90}


def matrix(key, J):
    n = len(PAIRS[key][0])
    M = np.ones((n, n))
    for p, v in J[key].items():
        i, j = map(int, p.split("-"))
        M[i, j], M[j, i] = float(v), 1.0 / float(v)
    return M


def fuzzy_weights(M, delta):
    """Buckley fuzzy geometric mean; TFN (max(1,a-d), a, a+d); centroid; normalised."""
    n = len(M)
    if n == 1:
        return np.array([1.0])
    L, U = np.ones((n, n)), np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = M[i, j]
            if a >= 1:
                l, u = max(1.0, a - delta), a + delta
            else:
                l, u = 1 / (1 / a + delta), 1 / max(1.0, 1 / a - delta)
            L[i, j], U[i, j] = l, u
            L[j, i], U[j, i] = 1 / u, 1 / l
    rl, rm, ru = (A.prod(1) ** (1 / n) for A in (L, M, U))
    d = (rl / ru.sum() + rm / rm.sum() + ru / rl.sum()) / 3
    return d / d.sum()


def fuzzy_detail(M, delta):
    """Rows of the fuzzy-AHP table: fuzzy weight (l, m, u), centroid and normalised weight."""
    n = len(M)
    if n == 1:
        return {"l": [1.0], "m": [1.0], "u": [1.0], "d": [1.0], "w": [1.0]}
    L, U = np.ones((n, n)), np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = M[i, j]
            if a >= 1:
                l, u = max(1.0, a - delta), a + delta
            else:
                l, u = 1 / (1 / a + delta), 1 / max(1.0, 1 / a - delta)
            L[i, j], U[i, j] = l, u
            L[j, i], U[j, i] = 1 / u, 1 / l
    rl, rm, ru = (A.prod(1) ** (1 / n) for A in (L, M, U))
    wl, wm, wu = rl / ru.sum(), rm / rm.sum(), ru / rl.sum()
    d = (wl + wm + wu) / 3
    return {"l": wl.tolist(), "m": wm.tolist(), "u": wu.tolist(), "d": d.tolist(), "w": (d / d.sum()).tolist()}


def ahp_detail(key, J, delta):
    """Crisp AHP block (matrix, weights, lambda max, CI, RI, CR) and the fuzzy table of one comparison matrix."""
    M = matrix(key, J)
    n = len(M)
    lam = float(max(np.linalg.eigvals(M).real)) if n > 1 else 1.0
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = consistency_ratio(M)
    gw = M.prod(1) ** (1 / n)
    return {"key": key, "names": PAIRS[key][0], "matrix": M.tolist(), "w": (gw / gw.sum()).tolist(), "lam": lam, "ci": float(ci),
            "ri": RI[n], "cr": cr, "fz": fuzzy_detail(M, delta)}


def consistency_ratio(M):
    n = len(M)
    if n < 3:
        return 0.0
    lam = max(np.linalg.eigvals(M).real)
    return float(((lam - n) / (n - 1)) / RI[n])


def weights(J, delta):
    g = fuzzy_weights(matrix("top", J), delta)
    loc = [fuzzy_weights(matrix("dem", J), delta), np.array([1.0]),
           fuzzy_weights(matrix("com", J), delta), fuzzy_weights(matrix("acc", J), delta)]
    return g, loc, np.concatenate(loc)


def final_w(gs, L):
    return np.array([gs[GO[j]] * L[j] for j in range(NC)])


def topsis(X, w, ben=None):
    ben = BEN if ben is None else ben
    R = X / np.sqrt((X ** 2).sum(0))
    V = R * w
    pis = np.where(ben, V.max(0), V.min(0))
    nis = np.where(ben, V.min(0), V.max(0))
    dp = np.sqrt(((V - pis) ** 2).sum(1))
    dn = np.sqrt(((V - nis) ** 2).sum(1))
    return dn / (dp + dn)


def saw(X, w):
    return (np.where(BEN, X / X.max(0), X.min(0) / X) * w).sum(1)


def group_weights(key, g, custom):
    if key == "S1":
        return np.array(g)
    p = np.array(custom if key == "C" else next(s[2] for s in SCEN if s[0] == key), float)
    return p / p.sum()


def sweep_w(g, L, k, val):
    gg = g.copy()
    others = [i for i in range(4) if i != k]
    gg[others] = g[others] / g[others].sum() * (1 - val)
    gg[k] = val
    return final_w(gg, L)


def switch_points(X, g, L, k, base_win):
    out = {}
    for d in (-1, 1):
        v = g[k]
        while 0.005 < v < 0.995:
            v += d * 0.001
            if int(np.argmax(topsis(X, sweep_w(g, L, k, v)))) != base_win:
                out[str(d)] = round(float(v), 3)
                break
    return out


def validate(vals):
    """Return a list of problems with the decision matrix (empty = OK)."""
    bad = []
    for j, c in enumerate(CRIT):
        for i in range(NA):
            v = vals[j][i]
            if not np.isfinite(v) or v <= 0 or v < c[6] or (c[7] is not None and v > c[7]):
                bad.append(f"{c[0]} {c[2]} / {ALT[i]}")
    return bad


def verification(vals, J, delta, custom, X, g, L, base, Cb, SW, allsc, crs):
    """Automatic checks: integrity, an independent plain-Python re-implementation and invariance tests."""
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "status": "OK" if ok else "CHECK", "detail": detail})

    add("Final weights sum to 1", abs(base.sum() - 1) < 1e-9, f"{base.sum():.6f}")
    add("Scenario group weights each sum to 1", all(abs(sum(s["gs"]) - 1) < 1e-9 for s in allsc), "all scenarios")
    lo = min(min(s["C"]) for s in allsc)
    hi = max(max(s["C"]) for s in allsc)
    add("TOPSIS closeness values lie in [0, 1]", lo >= 0 and hi <= 1, f"min {lo:.3f}, max {hi:.3f}")
    for k in ("top", "dem"):
        add(f"Consistency ratio of the {'main criteria' if k == 'top' else 'Demand'} matrix (limit 0.10)", crs[k] < 0.10, f"CR = {crs[k]:.3f}")
    # independent plain-Python implementation
    benefit = [bool(b) for b in BEN]
    sc_g = {s[0]: (None if s[2] is None else s[2]) for s in SCEN if s[0] not in ("C",)}
    runs = ind.full_run([list(map(float, r)) for r in vals], J, delta, GO, benefit, None,
                        {k: (lambda gg: gg) if v is None else [x / sum(v) for x in v] for k, v in sc_g.items()})
    dw = max(abs(a - b) for a, b in zip(runs["w"], base))
    dc = max(abs(a - b) for a, b in zip(runs["C"], Cb))
    ds = max(abs(a - b) for a, b in zip(runs["saw"], saw(X, base)))
    dsc = 0.0
    for s in allsc:
        if s["k"] in runs["scen"]:
            dsc = max(dsc, max(abs(a - b) for a, b in zip(runs["scen"][s["k"]], s["C"])))
    add("Independent implementation: final weights (8)", dw < 1e-9, f"max difference {dw:.1e}")
    add("Independent implementation: baseline closeness (4)", dc < 1e-9, f"max difference {dc:.1e}")
    add("Independent implementation: closeness of all scenarios", dsc < 1e-9, f"max difference {dsc:.1e}")
    add("Independent implementation: SAW scores (4)", ds < 1e-9, f"max difference {ds:.1e}")
    # delta = 0 equals crisp AHP (geometric mean)
    g0, _, L0 = weights(J, 0.0)
    crisp_g = matrix("top", J).prod(1) ** (1 / 4)
    crisp_g = crisp_g / crisp_g.sum()
    add("Fuzzy AHP with delta = 0 equals crisp AHP (geometric mean)", np.abs(g0 - crisp_g).max() < 1e-9, f"max difference {np.abs(g0 - crisp_g).max():.1e}")
    # scale invariance
    Xs = X * np.array([10, 0.1, 100, 3, 7, 0.5, 1000, 2])
    dsi = np.abs(topsis(Xs, base) - Cb).max()
    add("TOPSIS is unchanged when a criterion is rescaled (units test)", dsi < 1e-9, f"max difference {dsi:.1e}")
    # order invariance
    perm = [3, 1, 0, 2]
    dpi = np.abs(topsis(X[perm], base) - Cb[perm]).max()
    add("TOPSIS is unchanged when the alternatives are reordered", dpi < 1e-9, f"max difference {dpi:.1e}")
    return checks


def analyze(state):
    """Everything the dashboard needs, computed from the user's inputs."""
    vals = state.get("vals") or DEFAULT_VALS
    J = state.get("J") or DEFAULT_J
    delta = float(state.get("delta", 1.0))
    scen = state.get("scen", "S1")
    custom = state.get("custom") or [.25] * 4
    problems = validate(vals)
    if problems:
        return {"error": "Invalid input values: " + ", ".join(problems)}
    X = np.array(vals, float).T                      # alternatives x criteria
    g, loc, L = weights(J, delta)
    gs = group_weights(scen, g, custom)
    w = final_w(gs, L)
    C, SW = topsis(X, w), saw(X, w)
    base = final_w(g, L)
    Cb = topsis(X, base)
    base_win = int(np.argmax(Cb))
    allsc = []
    for k, name, _ in SCEN:
        if k == "C":
            continue
        gk = group_weights(k, g, custom)
        allsc.append({"k": k, "name": name, "gs": gk.tolist(), "C": topsis(X, final_w(gk, L)).tolist()})
    crs = {k: consistency_ratio(matrix(k, J)) for k in PAIRS}
    cg, cloc, cL = weights(J, 0.0)
    # sensitivity (baseline weights)
    sg = int(state.get("sensGroup", 0))
    xs = [round(v, 2) for v in np.arange(0.05, 0.9501, 0.05)]
    sweep = [topsis(X, sweep_w(g, L, sg, v)).tolist() for v in xs]
    facs = [0.5, 0.75, 1.25, 1.5]
    oat = []
    for j in range(NC):
        row = []
        for f in facs:
            ww = base.copy()
            ww[j] *= f
            row.append(int(np.argmax(topsis(X, ww / ww.sum()))))
        oat.append(row)
    wh = state.get("what") or {"alt": 2, "crit": 0, "pct": 20}
    Y = X.copy()
    Y[int(wh["alt"]), int(wh["crit"])] *= 1 + float(wh["pct"]) / 100
    changed = 0
    for i in range(NA):
        for j in range(NC):
            for f in (0.8, 1.2):
                Z = X.copy()
                Z[i, j] *= f
                changed += int(np.argmax(topsis(Z, base)) != base_win)
    deltas = []
    for d in (0, 0.5, 1, 1.5, 2, 3):
        gd, _, Ld = weights(J, d)
        Cd = topsis(X, final_w(gd, Ld))
        deltas.append({"delta": d, "demand": float(gd[0]), "C": Cd.tolist(), "win": int(np.argmax(Cd))})
    return {
        "engine": "Python / NumPy",
        "wt": {"g": g.tolist(), "L": L.tolist()}, "gs": gs.tolist(), "w": w.tolist(),
        "C": C.tolist(), "SW": SW.tolist(), "base": base.tolist(), "Cb": Cb.tolist(),
        "all": allsc, "crs": crs, "crispBase": final_w(cg, cL).tolist(),
        "ahp": {k: ahp_detail(k, J, delta) for k in PAIRS},
        "verify": verification(vals, J, delta, custom, X, g, L, base, Cb, SW, allsc, crs),
        "sens": {"group": sg, "xs": xs, "sweep": sweep, "switch": switch_points(X, g, L, sg, base_win),
                 "facs": facs, "oat": oat, "what": topsis(Y, base).tolist(),
                 "exh": {"runs": NA * NC * 2, "changed": changed}, "deltas": deltas},
    }


def meta():
    """Static description of the case (used by the dashboard to build its tables and controls)."""
    return {
        "alt": ALT, "groups": GROUPS,
        "crit": [{"id": c[0], "g": c[1], "name": c[2], "unit": c[3], "type": c[4], "vals": c[5], "min": c[6], "max": c[7]} for c in CRIT],
        "pairs": {k: {"names": v[0], "def": v[1]} for k, v in PAIRS.items()},
        "scen": [{"k": s[0], "name": s[1]} for s in SCEN],
    }


def web_meta_json():
    import json
    return json.dumps(meta())


def web_analyze_json(state_json):
    """Entry point used when the engine runs inside the browser (Pyodide on GitHub Pages)."""
    import json
    try:
        return json.dumps(analyze(json.loads(state_json)))
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
