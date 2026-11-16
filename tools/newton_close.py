"""Close a technique's cycle exactly with Newton's method on the starting layout.

With the split timings/aims fixed (from tools/solve_technique.py), one cycle maps a starting layout
L to an end layout G(L).  A periodic splitrun is a fixed point  G(L) = L  (up to rotation,
translation, reflection, and A<->B for mirror techniques).  Unknowns: positions of pieces 1..n-1
(piece 0 pinned at the origin).  Newton / Gauss-Newton with finite-difference Jacobians -- one
batched simulation of 2(n-1)+1 worlds per iteration -- converges in a handful of steps.

The Jacobian of G at the fixed point also gives the GEOMETRIC stability of the orbit: if its
spectral radius is < 1, position errors shrink every cycle (the geometric counterpart of the mass
stability rho).

    python tools/newton_close.py runs/techniques/random_e4_AABB_ranked.json 0
"""
import itertools, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa

TOTAL = 37000.0


def layout_vec(cells, swap, ref=None):
    """pieces ordered by (role, mass desc) -> (n,2) positions relative to piece 0; equal-mass ties are
    ordered to best match `ref` (if given)"""
    cs = sorted(cells, key=lambda c: ((c[1] ^ swap), -c[4]))
    P = np.array([[c[2], c[3]] for c in cs]); m = np.array([c[4] ** 2 / 100 for c in cs])
    roles = [c[1] ^ swap for c in cs]
    P = P - P[0]
    if ref is not None:
        # resolve ties (same role, masses within 1%) by the permutation closest to ref
        i = 1
        while i < len(cs):
            j = i
            while j + 1 < len(cs) and roles[j + 1] == roles[i] and abs(m[j + 1] - m[i]) < 0.01 * m[i]:
                j += 1
            if j > i:
                idx = list(range(i, j + 1))
                best = min(itertools.permutations(idx), key=lambda pm: np.linalg.norm(P[list(pm)] - ref[idx]))
                P[idx] = P[list(best)]
            i = j + 1
    return P, m, roles


def align(P, ref):
    """rotate (and maybe reflect) P about the origin to best match ref (least squares)"""
    best = None
    for refl in (1, -1):
        Q = P * np.array([1, refl])
        H = Q.T @ ref
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        A = Q @ R
        err = np.linalg.norm(A - ref)
        if best is None or err < best[0]:
            best = (err, A)
    return best[1]


def run_cycle(tech, V):
    """one cycle for each row of V; returns (end cells per row or None, cost)"""
    ends = {}
    c, cyc, _ = st.evaluate(tech, V, cycles=1, ends=ends)
    out = []
    for i in range(len(V)):
        e = ends.get(i)
        ok = e is not None and len(e) == tech.n
        out.append(e if ok else None)
    return out, c


def residual(tech, v, end_cells):
    L0 = np.vstack([[0, 0], v[:tech.npos].reshape(-1, 2)])
    swap = 1 if tech.mirror else 0
    P, m, roles = layout_vec(end_cells, swap, ref=L0)
    # the start layout's own order: rows() lists A pieces then B pieces in the catalogue order;
    # re-sort the start the same way (role, mass desc) so rows correspond
    start = tech.rows(v)
    S, _, _ = layout_vec([[i, r[0], r[1], r[2], r[3]] for i, r in enumerate(start)], 0)
    A = align(P, S)
    return (A - S)[1:].ravel(), S


def free_coords(tech):
    """all continuous parameters: starting positions, split cursors, movement cursors (not the gaps)"""
    idx = list(range(tech.npos))
    for e in range(tech.E):
        b = tech.npos + 7 * e
        idx += [b, b + 1, b + 3, b + 4, b + 5, b + 6]
    return np.array(idx)


def newton(tech, v, iters=15, h=4.0, log=print):
    """Gauss-Newton on ALL continuous parameters: the minimum-norm step that reduces the closing
    mismatch, so the solution stays where its eats still work."""
    v = v.copy()
    F = free_coords(tech)
    nf = len(F)
    for it in range(iters):
        V = np.repeat(v[None], nf + 1, 0)
        for j, c in enumerate(F):
            V[j + 1, c] += h
        ends, cost = run_cycle(tech, V)
        if ends[0] is None:
            log(f"   iter {it}: the cycle no longer completes cleanly (cost {cost[0]:.1f})")
            return v, None, False
        r0, S = residual(tech, v, ends[0])
        err = np.abs(r0).mean()
        log(f"   iter {it}: layout mismatch {err:.1f} units (one-cycle cost {cost[0]:.2f})")
        J = np.zeros((len(r0), nf))
        for j in range(nf):
            if ends[j + 1] is not None:
                J[:, j] = (residual(tech, V[j + 1], ends[j + 1])[0] - r0) / h
        if err < 3:
            # geometric stability: Jacobian of the return map w.r.t. the starting layout only
            G = J[:, :tech.npos] + np.eye(tech.npos)
            rho = max(abs(np.linalg.eigvals(G)))
            return v, rho, True
        step, *_ = np.linalg.lstsq(J, -r0, rcond=None)      # minimum-norm step
        for damp in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = v.copy()
            cand[F] += damp * step
            cand = np.clip(cand, tech.LO, tech.HI)
            e2, c2 = run_cycle(tech, cand[None])
            if e2[0] is not None and np.abs(residual(tech, cand, e2[0])[0]).mean() < err:
                v = cand
                break
        else:
            log("   no damped step improves: stuck")
            return v, None, False
    return v, None, False


if __name__ == "__main__":
    cat, k = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    techs = json.load(open(cat))
    tech = st.Technique(techs[k])
    base = os.path.basename(cat)[:-5]
    el = np.load(os.path.join(ROOT, "runs", "techniques", f"{base}_t{k}{tag}_p1.npy"))
    c1, _, _ = st.evaluate(tech, el, cycles=1)
    order = np.argsort(c1)
    print(f"technique {k} {tech.seq} {'MIRROR' if tech.mirror else 'IDENTITY'}: best one-cycle costs {np.round(c1[order[:5]], 2)}")
    t0 = time.time()
    for j in order[:6]:
        v, rho, ok = newton(tech, el[j])
        if ok:
            n, c, _ = st.chain_test(tech, v, 100)
            print(f"   closed (elite {j}); geometric stability {rho:.2f}; chain test {n}/100 ({time.time() - t0:.0f}s)")
            np.save(os.path.join(ROOT, "runs", "techniques", f"{base}_t{k}{tag}_newton.npy"), v[None])
            if n >= 100:
                break
