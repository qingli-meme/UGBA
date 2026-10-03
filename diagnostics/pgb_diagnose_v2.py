"""
pgb_diagnose_v2.py  —  fixes two bugs found in v1 diagnostic:

  BUG 1 (threshold ties): sparse BoW has many clean edges with cosine EXACTLY 0
         (e.g. Cora: 10.84% of clean edges), so a 'keep 90% clean' threshold is ill-defined
         and cos>=0 keeps 100%.  FIX: stop choosing a threshold. For every injected edge,
         report its CLEAN-ECDF PERCENTILE = fraction of clean edges it is MORE similar than.
         Tie-immune, scale-free, and it is exactly "how well this edge hides among real edges".
         A defender that keeps most clean edges must keep high-percentile injected edges.

  BUG 2 (mean prototype mismatch): v1 diagnostic measured cos(x_v, PURE prototype) and got
         0.801, but the harness v1 got 0.167. Two candidate causes, both isolated here:
           (i)  normalization of the prototype (sym-norm A_hat x vs row-norm D^-1 A x, self-loop or not);
           (ii) the harness injected edge uses prototype + delta*u (a CONTINUOUS offset on a
                sparse binary vector), which can collapse cosine.
         FIX: report, separately, cos for {pure prototype} and {prototype + delta*u}, for both
         normalizations, so the 0.801 -> 0.167 collapse is attributed to the right cause.

Usage:
  python pgb_diagnose_v2.py --root ./data --dataset Cora
  python pgb_diagnose_v2.py --root ./data --dataset Cora --delta 0.5    # match harness delta
  python pgb_diagnose_v2.py --root ./data --dataset Cora --trigger_npz real_edges.npz
"""
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected


def rowcos(a, b):
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data'); ap.add_argument('--dataset', default='Cora')
    ap.add_argument('--delta', type=float, default=0.5, help='continuous offset magnitude used by harness v1')
    ap.add_argument('--kw_list', type=int, nargs='+', default=[1, 3, 5, 10, 20])
    ap.add_argument('--n_victims', type=int, default=400)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--harness_thr', type=float, default=0.8)
    ap.add_argument('--trigger_npz', default='')
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed); torch.manual_seed(args.seed)

    data = Planetoid(args.root, args.dataset)[0]
    X = data.x.float(); y = data.y; N, d = X.shape
    ei = to_undirected(data.edge_index)
    r, c = ei; m = r < c; ci, cj = r[m], c[m]
    clean = rowcos(X[ci], X[cj]).numpy()
    clean_sorted = np.sort(clean)

    def clean_percentile(vals):
        """for each v in vals: fraction of clean edges with cosine <= v  (ECDF)."""
        return np.searchsorted(clean_sorted, vals, side='right') / len(clean_sorted)

    print(f"[data] {args.dataset} N={N} d={d} clean_edges={len(clean)} "
          f"zeros={100*np.mean(clean==0):.2f}%  clean cos mean={clean.mean():.3f} median={np.median(clean):.3f}")
    print(f"[Q1] clean survival @ {args.harness_thr:.2f} = {100*np.mean(clean>=args.harness_thr):.3f}%  "
          f"-> {'DEFENSE DESTROYS CLEAN GRAPH' if np.mean(clean>=args.harness_thr)<0.2 else 'ok'}")
    print("     (percentile view makes the threshold irrelevant; compare injected edges below)\n")

    # ---- prototypes, both normalizations ----
    Xn = X.numpy()
    adj = [[] for _ in range(N)]
    for a, b in zip(r.tolist(), c.tolist()):
        if a != b: adj[a].append(b)
    adj = [list(set(x)) for x in adj]
    deg = np.array([len(a) for a in adj], float)

    def proto_sym(v):   # A_hat x  (symmetric norm, self-loop) -- v1 diagnostic used this
        acc = Xn[v] / (deg[v] + 1.0)
        for u in adj[v]:
            acc = acc + Xn[u] / np.sqrt((deg[v] + 1.0) * (deg[u] + 1.0))
        return acc
    def proto_row(v):   # D^-1 A x  (row-normalized mean of neighbors, no self-loop)
        if not adj[v]: return Xn[v].copy()
        return np.mean([Xn[u] for u in adj[v]], axis=0)
    def proto_neighbor(v):
        cs = [float(Xn[v] @ Xn[u] / (np.linalg.norm(Xn[v]) * np.linalg.norm(Xn[u]) + 1e-12)) for u in adj[v]]
        return Xn[adj[v][int(np.argmax(cs))]]
    def proto_victim(v):
        return Xn[v].copy()

    victims = [v for v in range(N) if len(adj[v]) >= 2]
    rng.shuffle(victims); victims = victims[:args.n_victims]

    def edgecos(feat_fn):
        out = []
        for v in victims:
            f = feat_fn(v); xv = Xn[v]
            nf, nv = np.linalg.norm(f), np.linalg.norm(xv)
            out.append(0.0 if nf == 0 or nv == 0 else float(xv @ f / (nv * nf)))
        return np.array(out)

    def line(name, cosarr):
        pc = clean_percentile(cosarr)
        print(f"{name:<38}{cosarr.mean():>8.3f}{np.median(cosarr):>8.3f}{100*pc.mean():>12.1f}%{100*np.median(pc):>12.1f}%")

    hdr = f"{'strategy':<38}{'cosμ':>8}{'cos~':>8}{'clean-pctl μ':>12}{'pctl med':>12}"
    print("================ Q2: PURE prototype (no offset) =================")
    print(hdr); print('-'*len(hdr))
    line("mean prototype  [sym-norm A_hat x]", edgecos(proto_sym))
    line("mean prototype  [row-norm D^-1 A x]", edgecos(proto_row))
    line("neighbor-copy", edgecos(proto_neighbor))
    line("victim-copy (ceiling)", edgecos(proto_victim))

    # ---- isolate the delta*u offset effect (BUG 2 (ii)) ----
    print(f"\n================ Q2b: prototype + delta*u  (delta={args.delta}, u = random unit, v1-style) ===")
    print(hdr); print('-'*len(hdr))
    def with_offset(proto_fn):
        def f(v, _u):
            return proto_fn(v) + args.delta * _u
        return f
    for pname, pfn in [("sym-norm", proto_sym), ("row-norm", proto_row),
                       ("neighbor", proto_neighbor), ("victim", proto_victim)]:
        accs = []
        for _ in range(args.trials):
            u = rng.randn(d); u /= np.linalg.norm(u)
            cosv = np.array([
                (lambda fe, xv: 0.0 if np.linalg.norm(fe) == 0 else float(xv @ fe / (np.linalg.norm(xv) * np.linalg.norm(fe) + 1e-12)))
                (pfn(v) + args.delta * u, Xn[v]) for v in victims])
            accs.append(cosv)
        arr = np.concatenate(accs)
        line(f"{pname} + delta*u", arr)
    print("   >>> if these collapse vs Q2, the CONTINUOUS delta*u offset is the harness-0.167 culprit.")

    # ---- binary trigger words on victim-copy (the v2 proposal) ----
    print("\n================ Q3: victim-copy + k_w BINARY trigger words (v2 proposal) ========")
    print(hdr); print('-'*len(hdr))
    for kw in args.kw_list:
        accs = []
        for _ in range(args.trials):
            tw = rng.choice(d, size=kw, replace=False)
            cosv = []
            for v in victims:
                f = Xn[v].copy(); f[tw] = 1.0; xv = Xn[v]
                nf = np.linalg.norm(f)
                cosv.append(float(xv @ f / (np.linalg.norm(xv) * nf + 1e-12)))
            accs.append(np.array(cosv))
        line(f"victim-copy + {kw:>2d} words", np.concatenate(accs))
    print("   >>> watch where clean-pctl drops below ~50%: that is the stealth cost of distinctiveness.")

    if args.trigger_npz:
        z = np.load(args.trigger_npz)
        xx = torch.tensor(z['x']).float(); ee = torch.tensor(z['edges']).long()
        tc = rowcos(xx[ee[0]], xx[ee[1]]).numpy()
        print("\n================ REAL harness trigger edges =====================")
        line("harness trigger edges", tc)

    print("\n================ READOUT ========================================")
    print("A) Q2 rows disagree across normalizations -> fix base_mean in harness to match intended norm.")
    print("B) Q2 pure prototype pctl high BUT Q2b (with delta*u) collapses -> the delta*u offset is the bug;")
    print("   drop continuous offset, use Q3 binary words. (confirms v2 design.)")
    print("C) Q3: if victim-copy + ENOUGH words to be learnable keeps clean-pctl high -> prototype was the bug,")
    print("   run passband_trigger_v2 ASR. If clean-pctl collapses with few words -> paradigm wall, write negative result.")


if __name__ == '__main__':
    main()
