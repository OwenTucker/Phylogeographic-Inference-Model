import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

def extract_cbg(original_id: str) -> str:
    return original_id.rsplit("-", 1)[0]

def parse_true_tree(tree_path: Path) -> dict[str, str]:
    cbg_of: dict[str, str] = {}
    with open(tree_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "," not in line:
                continue
            left, _ = line.rsplit(",", 1)
            node_id = left.split(".")[-1]
            cbg_of[node_id] = extract_cbg(node_id)
    return cbg_of

def build_q_matrix(mobility_path: Path,
                   cbg_list: list[str]) -> np.ndarray:
    cbg_index = {cbg: i for i, cbg in enumerate(cbg_list)}
    n = len(cbg_list)
    raw = np.zeros((n, n), dtype=np.float64)

    with open(mobility_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src_cbg = row["poi_cbg_source"].strip()
            dst_cbg = row["poi_cbg_destination"].strip()
            if src_cbg not in cbg_index or dst_cbg not in cbg_index:
                continue
            if src_cbg == dst_cbg:
                continue
            i, j = cbg_index[src_cbg], cbg_index[dst_cbg]
            raw[i, j] += float(row["src_prob"]) * float(row["des_prob"])
    Q = np.log1p(raw)
    for i in range(n):
        row_sum = Q[i].sum()
        if row_sum > 0:
            Q[i] /= row_sum
        else:
            Q[i] = 1.0 / (n - 1)
            Q[i, i] = 0.0
        Q[i, i] = -Q[i].sum()   # row sums to 0

    return Q

def compute_mobility_prior(mobility_path: Path, cbg_list: list[str]) -> np.ndarray:
    cbg_index = {cbg: i for i, cbg in enumerate(cbg_list)}
    n   = len(cbg_list)
    pri = np.zeros(n)

    with open(mobility_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dst = row["poi_cbg_destination"].strip()
            if dst in cbg_index:
                pri[cbg_index[dst]] += float(row["des_prob"])

    pri = pri + 1e-6
    return pri / pri.sum()


def compute_stationary(Q: np.ndarray) -> np.ndarray:
    n = len(Q)
    A = Q.T.copy()
    A[-1, :] = 1.0          
    b = np.zeros(n)
    b[-1] = 1.0             

    try:
        pi = np.linalg.solve(A, b)
        pi = np.clip(pi, 0, None)
        s  = pi.sum()
        if s > 0:
            return pi / s
    except np.linalg.LinAlgError:
        pass

    _, _, Vt = np.linalg.svd(Q.T)
    pi = np.abs(Vt[-1])    
    pi = np.clip(pi, 0, None)
    s  = pi.sum()
    if s == 0:
        return np.ones(n) / n
    return pi / s

class TransitionCache:
    def __init__(self, Q: np.ndarray, precision: int = 4):
        self.Q         = Q
        self.precision = precision
        self._cache: dict[float, np.ndarray] = {}

    def get(self, branch_length: float) -> np.ndarray:
        t = round(branch_length, self.precision)
        if t not in self._cache:
            self._cache[t] = expm(self.Q * t)
            P = self._cache[t]
            P = np.clip(P, 0, None)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self._cache[t] = P / row_sums
        return self._cache[t]

    @property
    def cache_size(self) -> int:
        return len(self._cache)

def run_belief_propagation(nodes: dict,edges: list,cbg_index: dict[str, int],cache: TransitionCache,latent_prior: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n_cbg = len(cbg_index)
    parent_edge:    dict[str, tuple[str, float]] = {}
    children_edges: dict[str, list[tuple[str, float]]] = {nid: [] for nid in nodes}

    for e in edges:
        p, c, bl = e["parent_id"], e["child_id"], e["branch_length"]
        parent_edge[c] = (p, bl)
        children_edges[p].append((c, bl))

   
    default_latent = latent_prior if latent_prior is not None else np.ones(n_cbg) / n_cbg
    evidence: dict[str, np.ndarray] = {}
    for nid, node in nodes.items():
        if node["type"] == "observed" and node["cbg"] in cbg_index:
            phi = np.zeros(n_cbg)
            phi[cbg_index[node["cbg"]]] = 1.0
        else:
            phi = default_latent.copy()
        evidence[nid] = phi

    
    root_ids = [nid for nid, node in nodes.items() if node["parent"] is None]

    order = []
    queue = list(root_ids)
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for cid, _ in children_edges[nid]:
            queue.append(cid)

    
    mu_up: dict[str, np.ndarray] = {}

    for nid in reversed(order):
        
        msg = evidence[nid].copy()

        for cid, bl in children_edges[nid]:
            P = cache.get(bl)   # P[i,j] = P(child=j | parent=i)
            child_msg = mu_up[cid]   # message at child node
            # marginalise over child state: sum_j P[i,j] * child_msg[j]
            msg = msg * (P @ child_msg)

        s = msg.sum()
        mu_up[nid] = msg / s if s > 0 else np.ones(n_cbg) / n_cbg

    # mu_down[nid] = message node receives from above (shape: n_cbg)
    mu_down: dict[str, np.ndarray] = {}
    for rid in root_ids:
        mu_down[rid] = np.ones(n_cbg)   # roots have no parent 

    for nid in order:
        p_msg = mu_down[nid]
        local = evidence[nid]

        for cid, bl in children_edges[nid]:
            P = cache.get(bl)
            parent_excl = p_msg * local
            for other_cid, other_bl in children_edges[nid]:
                if other_cid == cid:
                    continue
                P_other = cache.get(other_bl)
                parent_excl = parent_excl * (P_other @ mu_up[other_cid])
            down_msg = P.T @ parent_excl
            s = down_msg.sum()
            mu_down[cid] = down_msg / s if s > 0 else np.ones(n_cbg) / n_cbg

    
    marginals: dict[str, np.ndarray] = {}
    for nid in nodes:
        belief = mu_up[nid] * mu_down[nid]
        s = belief.sum()
        marginals[nid] = belief / s if s > 0 else np.ones(n_cbg) / n_cbg

    return marginals

def evaluate(marginals: dict[str, np.ndarray],
             nodes: dict,
             true_cbg_of: dict[str, str],
             cbg_list: list[str]) -> dict:
   
    cbg_index = {cbg: i for i, cbg in enumerate(cbg_list)}
    n_cbg = len(cbg_list)
    true_counts = np.zeros(n_cbg)
    for node_id, cbg in true_cbg_of.items():
        if cbg in cbg_index:
            true_counts[cbg_index[cbg]] += 1
    true_total = true_counts.sum()
    true_dist  = true_counts / true_total if true_total > 0 else true_counts

    recon_counts = np.zeros(n_cbg)
    for marg in marginals.values():
        recon_counts += marg
    recon_total = recon_counts.sum()
    recon_dist  = recon_counts / recon_total if recon_total > 0 else recon_counts

    l1 = float(np.abs(true_dist - recon_dist).sum())

    eps = 1e-12
    kl = float(np.sum(
        true_dist[true_dist > 0] *
        np.log(true_dist[true_dist > 0] / (recon_dist[true_dist > 0] + eps))
    ))

    # Argmax accuracy on observed nodes with known ground truth
    n_correct = 0
    n_scored  = 0
    for nid, node in nodes.items():
        if node["type"] != "observed":
            continue
        orig = node.get("original_id")
        if orig is None:
            continue
        true_cbg = true_cbg_of.get(orig.split("-")[0] + "-" + orig.split("-")[1]
                                   if "-" in orig else orig)
        # Try direct lookup by original_id node key
        true_cbg = true_cbg_of.get(orig)
        if true_cbg is None:
            continue
        pred_cbg = cbg_list[int(np.argmax(marginals[nid]))]
        n_scored  += 1
        n_correct += int(pred_cbg == true_cbg)

    accuracy = n_correct / n_scored if n_scored > 0 else float("nan")

    return {
        "true_counts":  true_counts,
        "recon_counts": recon_counts,
        "true_dist":    true_dist,
        "recon_dist":   recon_dist,
        "l1_distance":  l1,
        "kl_divergence": kl,
        "argmax_accuracy": accuracy,
        "n_scored":     n_scored,
        "n_correct":    n_correct,
        "true_total":   int(true_total),
        "recon_total":  int(recon_total),
    }


# ── Writers ───────────────────────────────────────────────────────────────────

def write_outputs(marginals: dict[str, np.ndarray],
                  nodes: dict,
                  cbg_list: list[str],
                  eval_result: dict,
                  input_dir: Path,
                  sim_name: str,
                  latent_prior_mode: str = "uniform"):

    # bp_marginals.csv — full marginal vector per node
    with open(input_dir / "bp_marginals.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "type", "cbg_true"] + cbg_list)
        for nid, node in nodes.items():
            true_cbg = node.get("cbg") or ""
            row = [nid, node["type"], true_cbg] + marginals[nid].tolist()
            writer.writerow(row)

    # bp_predictions.csv — argmax prediction per node
    with open(input_dir / "bp_predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "node_id", "type", "cbg_true", "cbg_pred", "pred_prob", "correct"
        ])
        writer.writeheader()
        for nid, node in nodes.items():
            marg     = marginals[nid]
            pred_idx = int(np.argmax(marg))
            pred_cbg = cbg_list[pred_idx]
            true_cbg = node.get("cbg") or ""
            correct  = int(pred_cbg == true_cbg) if true_cbg else ""
            writer.writerow({
                "node_id":   nid,
                "type":      node["type"],
                "cbg_true":  true_cbg,
                "cbg_pred":  pred_cbg,
                "pred_prob": round(float(marg[pred_idx]), 6),
                "correct":   correct,
            })

    # bp_evaluation.csv — per-CBG true vs reconstructed counts
    with open(input_dir / "bp_evaluation.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "cbg", "true_count", "recon_count", "true_freq", "recon_freq"
        ])
        writer.writeheader()
        for i, cbg in enumerate(cbg_list):
            writer.writerow({
                "cbg":        cbg,
                "true_count":  round(eval_result["true_counts"][i],  4),
                "recon_count": round(eval_result["recon_counts"][i], 4),
                "true_freq":   round(eval_result["true_dist"][i],    6),
                "recon_freq":  round(eval_result["recon_dist"][i],   6),
            })

    # bp_summary.txt — scalar metrics
    with open(input_dir / "bp_summary.txt", "w") as fh:
        fh.write(f"Simulation     : {sim_name}\n")
        fh.write(f"Latent prior   : {latent_prior_mode}\n")
        fh.write(f"True nodes     : {eval_result['true_total']}\n")
        fh.write(f"BP nodes       : {eval_result['recon_total']:.1f}\n")
        fh.write(f"CBGs in state  : {len(cbg_list)}\n")
        fh.write(f"L1 distance    : {eval_result['l1_distance']:.6f}\n")
        fh.write(f"KL divergence  : {eval_result['kl_divergence']:.6f}\n")
        fh.write(f"Argmax accuracy: {eval_result['argmax_accuracy']:.4f}"
                 f"  ({eval_result['n_correct']}/{eval_result['n_scored']} observed)\n")


# ── Per-simulation entry ──────────────────────────────────────────────────────

def process_simulation(sim_dir: Path, mobility_path: Path, latent_prior_mode: str = "uniform", label: str = "") -> bool:
    input_dir  = sim_dir / (f"outbreaker_input_{label}" if label else "outbreaker_input")
    tree_path  = sim_dir / "tree"
    json_path  = input_dir / "full_tree.json"

    for p in (json_path, tree_path):
        if not p.exists():
            print(f"  [ERROR] {sim_dir.name}: missing {p.name}")
            return False

    try:
        # ── load tree ─────────────────────────────────────────────────────────
        with open(json_path) as fh:
            tree_data = json.load(fh)

        nodes = {n["node_id"]: n for n in tree_data["nodes"]}
        edges = tree_data["edges"]

        # ── CBG state space: union of sampled CBGs and all true tree CBGs ──────
        # This ensures BP can assign probability to CBGs that exist in the true
        # outbreak but were never sampled — critical for unbiased reconstruction.
        sampled_cbgs = {n["cbg"] for n in nodes.values() if n["cbg"] is not None}
        true_cbg_of  = parse_true_tree(tree_path)
        true_cbgs    = set(true_cbg_of.values())
        unobserved_cbgs = true_cbgs - sampled_cbgs

        cbg_set  = sampled_cbgs | true_cbgs
        cbg_list = sorted(cbg_set)
        cbg_index = {cbg: i for i, cbg in enumerate(cbg_list)}
        n_cbg = len(cbg_list)

        print(f"\n  {sim_dir.name}")
        print(f"    Nodes            : {len(nodes)}  ({sum(1 for n in nodes.values() if n['type']=='observed')} observed, "
              f"{sum(1 for n in nodes.values() if n['type']=='latent')} latent)")
        print(f"    CBG state space  : {n_cbg}  "
              f"(sampled={len(sampled_cbgs)}  true-only={len(unobserved_cbgs)})")

        # ── kappa diagnostics ─────────────────────────────────────────────────
        all_kappas = [e["kappa"] for e in edges if e["kappa"] > 0]
        if all_kappas:
            print(f"    Kappa (non-zero) : mean={sum(all_kappas)/len(all_kappas):.2f}  "
                  f"max={max(all_kappas)}  "
                  f"edges_with_kappa={len(all_kappas)}/{len(edges)}")
        else:
            print(f"    Kappa            : all zero (no latent nodes)")

        # ── build Q ───────────────────────────────────────────────────────────
        print(f"    Building Q matrix...")
        Q = build_q_matrix(mobility_path, cbg_list)
        print(f"    Q: {n_cbg}x{n_cbg}  sparsity={(Q == 0).sum() / Q.size:.1%}")

        # ── stationary distribution ───────────────────────────────────────────
        pi = compute_stationary(Q)
        top5_idx = np.argsort(pi)[::-1][:5]
        print(f"    Stationary dist  : top-5 CBGs by π")
        for idx in top5_idx:
            print(f"      {cbg_list[idx]:15s}  π={pi[idx]:.4f}")
        print(f"      Entropy: {-np.sum(pi * np.log(pi + 1e-12)):.3f}  "
              f"(max={np.log(n_cbg):.3f}  ratio={(-np.sum(pi * np.log(pi + 1e-12)))/np.log(n_cbg):.3f})")

        # ── transition matrix cache ───────────────────────────────────────────
        cache = TransitionCache(Q)

        # ── latent prior ──────────────────────────────────────────────────────
        if latent_prior_mode == "stationary":
            latent_prior = pi
            print(f"    Latent prior     : stationary distribution")
        elif latent_prior_mode == "empirical":
            # CBG frequency distribution of observed (sampled) nodes
            emp = np.zeros(n_cbg)
            for node in nodes.values():
                if node["type"] == "observed" and node["cbg"] in cbg_index:
                    emp[cbg_index[node["cbg"]]] += 1
            s = emp.sum()
            latent_prior = emp / s if s > 0 else np.ones(n_cbg) / n_cbg
            top3 = np.argsort(latent_prior)[::-1][:3]
            print(f"    Latent prior     : empirical (sampled CBG distribution)")
            print(f"      top-3: " + "  ".join(
                f"{cbg_list[i]}={latent_prior[i]:.3f}" for i in top3))
        elif latent_prior_mode == "mobility":
            mob = compute_mobility_prior(mobility_path, cbg_list)
            latent_prior = mob
            top3 = np.argsort(latent_prior)[::-1][:3]
            print(f"    Latent prior     : mobility (destination marginals)")
            print(f"      top-3: " + "  ".join(
                f"{cbg_list[i]}={latent_prior[i]:.3f}" for i in top3))
        else:
            latent_prior = None
            print(f"    Latent prior     : uniform")

        # ── run BP ────────────────────────────────────────────────────────────
        print(f"    Running belief propagation...")
        marginals = run_belief_propagation(nodes, edges, cbg_index, cache, latent_prior)
        print(f"    Cache entries used: {cache.cache_size}")

        # ── ground truth (already parsed above for state space) ──────────────
        print(f"    True tree nodes  : {len(true_cbg_of)}")

        # ── evaluate ──────────────────────────────────────────────────────────
        eval_result = evaluate(marginals, nodes, true_cbg_of, cbg_list)
        print(f"    L1 distance      : {eval_result['l1_distance']:.4f}")
        print(f"    KL divergence    : {eval_result['kl_divergence']:.4f}")
        print(f"    Argmax accuracy  : {eval_result['argmax_accuracy']:.4f}"
              f"  ({eval_result['n_correct']}/{eval_result['n_scored']})")

        # ── write outputs ─────────────────────────────────────────────────────
        write_outputs(marginals, nodes, cbg_list, eval_result,
                      input_dir, sim_dir.name, latent_prior_mode)
        print(f"    Written          : bp_marginals.csv, bp_predictions.csv,")
        print(f"                       bp_evaluation.csv, bp_summary.txt")

        return True

    except Exception as e:
        import traceback
        print(f"  [ERROR] {sim_dir.name}: {e}")
        traceback.print_exc()
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CTMC + Belief Propagation inference on transmission tree"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Single simulation directory")
    group.add_argument("--all", type=Path, help="Parent directory of all sim dirs")
    parser.add_argument("--mobility", type=Path, required=True,
                        help="Path to cbg2cbg.csv mobility file")
    parser.add_argument("--latent-prior", type=str, default="uniform",
                        choices=["uniform", "empirical", "mobility"],
                        help="Prior for latent nodes: "
                             "'uniform' (no information), "
                             "'empirical' (sampled CBG distribution), or "
                             "'mobility' (destination marginals from cbg2cbg, default: uniform)")
    parser.add_argument("--label", type=str, default="",
                        help="Input/output folder suffix, e.g. '10pct' -> outbreaker_input_10pct/")
    args = parser.parse_args()

    if not args.mobility.exists():
        sys.exit(f"[ERROR] Mobility file not found: {args.mobility}")

    if args.dir:
        dirs = [args.dir]
    else:
        parent = args.all
        if not parent.is_dir():
            sys.exit(f"[ERROR] Not a directory: {parent}")
        input_folder = f"outbreaker_input_{args.label}" if args.label else "outbreaker_input"
        dirs = sorted(
            d for d in parent.iterdir()
            if d.is_dir() and (d / input_folder / "full_tree.json").exists()
        )
        print(f"Found {len(dirs)} simulation directories with full_tree.json\n")

    success = 0
    for sim_dir in dirs:
        ok = process_simulation(sim_dir, args.mobility, args.latent_prior, args.label)
        if ok:
            success += 1

    print(f"\nDone: {success}/{len(dirs)} directories processed successfully.")


if __name__ == "__main__":
    main()