# Implementation Guide — Right-Subtree Collapse for L0-penalised BnB

**Audience:** an AI coding assistant adding **one** optimisation to a Branch-and-Bound solver that
already solves an **ℓ₀-penalised** problem of the form

```
minimize   f(x) + β·‖x‖₀        subject to   x feasible (e.g. ‖x‖₂² ≤ r)
```

by branching on which coordinates are in the support.

**Scope — read carefully.** You are implementing the **right-subtree collapse** and **nothing
else**. It is exactly **one** conditional added where the right child is enqueued. Do **not** add
any other pruning, do **not** touch the left child, the branching rule, the priority order, the
relaxation solver, or the termination condition. It is a **pure speedup** — the returned optimum
must be bit-for-bit identical to before. On a reference implementation it removed ~11% of nodes
(node ratio ≈ 0.887) with **100% identical optima**.

The correctness argument is subtle (the node lower bound is a "lower score", not a classical
subtree bound). Read §3 before editing; do not "simplify" it away.

---

## 0. Confirm your BnB matches this structure

The collapse is only valid if your solver has ALL of the following. **Check each against your code
and map the names.** If any fails, STOP and report it instead of forcing the change in.

- Each node is a pair `(S, P)`:
  - `S` = coordinates already **forced into** the support (added by prior LEFT branches).
  - `P` = coordinates still **free** to branch on. `supp(x) ⊆ S ∪ P` for anything reachable.
- Node **lower bound** is `lb(S,P) = relax(S ∪ P) + β·|S|`, where `relax(I)` is the optimal value
  of your continuous subproblem restricted to coordinates `I`. **The `+ β·|S|` term is essential** —
  the collapse relies on it.
- Branching on `i ∈ P` creates:
  - **LEFT child** `(S ∪ {i}, P\{i})` — leave this completely unchanged.
  - **RIGHT child** `(S, P\{i})` with `lb_right = relax(S ∪ (P\{i})) + β·|S|`.
- The tree is explored **best-first**: a priority queue ordered by `lb` ascending. Termination is
  `if global_ub - lb < relErr: break`.
- `global_ub` is the incumbent (best feasible objective found so far, includes its `β·‖·‖₀`).
- When a node is popped, an **incumbent update** runs first:
  `if node.ub - global_ub < relErr: global_ub = node.ub; record node's solution`.
  For a node whose support is exactly `S`, `node.ub` equals `relax(S) + β·|S|` — i.e. **the value of
  the support-`S` solution is already applied as an incumbent when the node is popped.** This fact
  is what makes the collapse lossless (see §3).

If your `lb` omits `+ β·|S|`, or the search is not best-first, STOP and report that the
preconditions don't hold.

---

## 1. The change (one conditional, right child only)

Your baseline, after solving the right child's relaxation to get
`lb_right = relax(S ∪ P') + β·|S|` (where `P' = P \ {i}`), currently enqueues:

```
enqueue(lb = lb_right, ub = <parent ub>, S = S, P = P', ...)
```

**Replace that single `enqueue` with this `if/else`:**

```
if lb_right + β >= global_ub - relErr:
    enqueue(lb = lb_right, ub = <parent ub>, S = S, P = [] , ...)   # COLLAPSE: leaf, P empty
else:
    enqueue(lb = lb_right, ub = <parent ub>, S = S, P = P', ...)   # unchanged baseline
```

That is the entire change. Setting `P = []` makes the node a **leaf**: when it is later popped it
does its incumbent update and then stops (no branching, because `P` is empty). That one leaf
replaces the whole right subtree, which otherwise costs `O(|P|)` nodes.

- `β` is the ℓ₀ penalty coefficient; `relErr` is your existing numerical tolerance (e.g. `1e-8`) —
  use the SAME one already in the code.
- The right child still **inherits the parent's `ub`** exactly as before. Do not invent a new `ub`.
- If `lb_right ≥ global_ub`, the collapse still just enqueues a leaf; it will be handled by the
  normal incumbent/`break` logic when popped. No special case needed.

---

## 2. Reference snippet (adapt identifiers to your code)

```python
# ... node popped: lb, ub, S, P, x_parent unpacked; incumbent update, break, leaf checks done ...
# choose branching variable i ∈ P (your existing rule), then:
P_prime = [j for j in P if j != i]

# ---- RIGHT CHILD (the ONLY place that changes) ----
if len(S) + len(P_prime) >= 1:
    w = S + P_prime
    x_r, relax_r = solve_relaxation(w)          # your existing relaxation solver on submatrix w
    lb_right = relax_r + beta * len(S)

    if lb_right + beta >= global_ub - relErr:
        enqueue(lb_right, ub, S, [],      x_parent)   # COLLAPSE to leaf
    else:
        enqueue(lb_right, ub, S, P_prime, x_parent)   # baseline (unchanged)

# ---- LEFT CHILD: leave EXACTLY as it was in the baseline ----
# ... unchanged ...
```

Do not change `solve_relaxation`, the priority key, the branching rule, or the `break`.

---

## 3. Why it is correct (do not skip — it is counter-intuitive)

**The node bound `lb(S,P)` is a "lower score", NOT a classical subtree lower bound.** The relaxation
optimises *freely* over `S ∪ P`; it may return `x_j = 0` for some `j ∈ S`. So a feasible point
reachable from the node can have objective *below* `lb`. Example (`S={1,2,3}, P={4,5}`): a solution
supported on `{1,2}` has objective `relax({1,2}) + 2β`, which can be **less** than
`lb = relax({1,2,3,4,5}) + 3β`. So you **cannot** justify pruning by "lb bounds everything below this
node" — that is false. Correctness rests on the **canonical / optimal-path** argument:

- Call `x` *canonical* at `(S,P)` if `S ⊆ supp(x) ⊆ S ∪ P`. The tree partitions supports so every
  non-canonical point is canonical for a **different** ancestor's right subtree and is found there —
  never lost.
- **Lemma:** for canonical `x̄`, `f(x̄) ≥ relax(S∪P) + β|S| = lb`. (supp ⊆ S∪P gives the relax term;
  `S ⊆ supp` gives `‖x̄‖₀ ≥ |S|`.)
- **Optimal-path theorem:** the global optimum `x*` is canonical at every node on its own root→leaf
  path (LEFT on `j ∈ supp(x*)`, RIGHT on `j ∉ supp(x*)`), so `lb ≤ f(x*) ≤ global_ub` there — **its
  path is never pruned.** BnB stays exact.

**Why the collapse is safe.** Any canonical solution in the right subtree using `k ≥ 1` coordinate
from `P'` has `‖x‖₀ ≥ |S| + k` and is canonical at the right child, so
`f(x) ≥ lb_right + β·k ≥ lb_right + β ≥ global_ub`. None can strictly improve the incumbent. The only
remaining right-subtree candidate uses **zero** coordinates from `P'`, i.e. support exactly `S`,
worth `relax(S) + β|S|` — and that value **was already applied as this node's own incumbent `ub`
when it was popped** (see §0, last bullet). So collapsing the right subtree into a single leaf loses
nothing, and the returned optimum is unchanged.

---

## 4. Hard safety constraints (never violate)
- **Never weaken** the termination `if global_ub - lb < relErr: break`.
- `lb` must remain a **real relaxation value** (`relax(...) + β|S|`); never inflate it.
- Only **tighten** (lower) `global_ub` via genuine feasible solutions. The collapse adds pruning
  only — it must never raise `lb` or lower `global_ub` artificially.
- Change **only** the right-child enqueue. Leave the left child, branching, ordering, and the
  relaxation solver untouched.

## 5. How to verify (mandatory, after implementing)
1. **Correctness gate:** on ≥100 random instances spanning your problem sizes, run the solver
   **with and without** the collapse. Assert the returned objective is **identical** (`|Δ| < 1e-6`)
   on every instance, and that the returned `x` achieves it. Any mismatch = bug; revert and
   re-check the criterion (`lb_right + β ≥ global_ub`, the `+ β`, and `relErr`).
2. **Benefit check:** report node-count ratio (with/without). Expect roughly **0.85–0.92** (≈10–15%
   fewer nodes); larger `β` and tighter incumbents collapse more.
3. **Edge cases:** `β` large (collapses fire often), `β` near 0 (collapse almost never fires — must
   still match), optima with full support, and the empty-support case.

## 6. Common pitfalls
- **Wrong direction of the criterion.** It is `lb_right + β ≥ global_ub` (with `relErr` slack).
  A sign error prunes the optimal path → wrong answer; the §5 gate catches it.
- **Forgetting `+ β·|S|` in `lb`.** If your `lb` omits the penalty term, the collapse is invalid.
- **Using a stale `global_ub`.** Evaluate the criterion against the current incumbent.
- **Changing anything besides the right-child enqueue.** Don't. The scope is one conditional.

---

*Provenance:* the "right-subtree collapse / reduced-cost fixing" from a sparse trust-region BnB
project; full correctness proof in that project's `tree_idea2_collapse_proof.md`. The same BnB
skeleton (node `(S,P)`, `lb = relax(S∪P)+β|S|`, best-first, LEFT/RIGHT children) appears in the
Şen–Akkaya–Pınar sparse mean–variance portfolio BnB — if your solver is that family, the mapping is
direct.
