import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run_bp(scripts_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "run_bp", scripts_dir / "run_bp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def entropy(freq_vec: np.ndarray) -> float:
    """Shannon entropy of a probability vector in nats."""
    p = freq_vec[freq_vec > 0]
    return float(-np.sum(p * np.log(p)))

def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Cross-entropy H(p, q) = -sum_i p_i * log(q_i), computed over p > 0.
    Decomposes as H(p) + KL(p||q): lower means q better approximates p.
    """
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask] + eps)))

GEORGIA_CENTROIDS: dict[str, tuple[float, float]] = {
    "13021": (32.8065, -83.6974), "13029": (31.9891, -81.4368),
    "13031": (32.3957, -81.7432), "13039": (30.8955, -81.6404),
    "13043": (33.5813, -85.0757), "13045": (34.9056, -85.1390),
    "13049": (32.0002, -81.0936), "13057": (34.2424, -84.4718),
    "13059": (33.9577, -83.3689), "13063": (33.5423, -84.3574),
    "13067": (33.9415, -84.5762), "13071": (33.5389, -82.2103),
    "13073": (31.1534, -83.4291), "13075": (33.3570, -84.7627),
    "13077": (32.7154, -83.9791), "13085": (30.8834, -84.5766),
    "13089": (33.7724, -84.2251), "13095": (33.6990, -84.7654),
    "13097": (31.3249, -84.8919), "13101": (32.3591, -81.3455),
    "13103": (34.1134, -82.8593), "13111": (33.4112, -84.4952),
    "13113": (34.2654, -85.2135), "13115": (34.2248, -84.1276),
    "13119": (34.3722, -83.2374), "13121": (33.7957, -84.4659),
    "13127": (34.5007, -84.8791), "13133": (33.9609, -84.0219),
    "13135": (34.6269, -83.5314), "13137": (34.3159, -83.8162),
    "13139": (33.2680, -83.0049), "13143": (32.7350, -84.9087),
    "13145": (34.3491, -82.9700), "13149": (33.4496, -84.1571),
    "13151": (32.4617, -83.6508), "13153": (31.5982, -83.2760),
    "13155": (34.1332, -83.5625), "13157": (33.3215, -83.6882),
    "13163": (32.7974, -81.9696), "13167": (33.0312, -83.5588),
    "13173": (32.4641, -82.9231), "13175": (31.7890, -84.1401),
    "13177": (31.8084, -81.4725), "13179": (33.7919, -82.4700),
    "13181": (31.7494, -81.7279), "13183": (30.8345, -83.2686),
    "13189": (31.5049, -81.3913), "13193": (34.1275, -83.2113),
    "13199": (31.1602, -84.7321), "13205": (32.1716, -82.5307),
    "13211": (32.5100, -84.8771), "13213": (33.5503, -83.8534),
    "13215": (33.8288, -83.4345), "13219": (33.9222, -84.8798),
    "13221": (32.5698, -83.8306), "13223": (34.4650, -84.4713),
    "13225": (31.3559, -82.2152), "13227": (33.0868, -84.3829),
    "13241": (33.3682, -82.0741), "13243": (33.6558, -84.0207),
    "13251": (33.2596, -84.2917), "13253": (34.5596, -83.3001),
    "13255": (32.0692, -84.8199), "13257": (32.0390, -84.1914),
    "13271": (30.8723, -83.9165), "13273": (31.4601, -83.5234),
    "13275": (32.1252, -82.3287), "13281": (33.0326, -85.0282),
    "13285": (32.6651, -83.4306), "13287": (34.8344, -83.9976),
    "13289": (32.8819, -84.2896), "13291": (34.7248, -85.2962),
    "13293": (33.7733, -83.7268), "13295": (31.0541, -82.4095),
    "13297": (33.4077, -82.6860), "13301": (31.5540, -81.9023),
    "13305": (32.1168, -82.7278), "13307": (34.6479, -83.7406),
    "13309": (34.7971, -84.9702), "13315": (32.8049, -83.1595),
}


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
   
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def build_cost_matrix(cbg_list: list[str]) -> np.ndarray | None:
    
    if not HAS_OT:
        return None
    n = len(cbg_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            c1 = GEORGIA_CENTROIDS.get(cbg_list[i][:5])
            c2 = GEORGIA_CENTROIDS.get(cbg_list[j][:5])
            d  = haversine_km(c1[0], c1[1], c2[0], c2[1]) if (c1 and c2) else 500.0
            M[i, j] = M[j, i] = d
    return M


def earth_mover_distance(p: np.ndarray, q: np.ndarray,
                         M: np.ndarray | None) -> float | None:
   
    if not HAS_OT or M is None:
        return None
    if p.sum() == 0 or q.sum() == 0:
        return None   # degenerate distribution (e.g. no population data matched)
    p = p / p.sum()
    q = q / q.sum()
    return float(ot.emd2(p.astype(np.float64),
                          q.astype(np.float64),
                          M.astype(np.float64)))



def parse_true_tree(tree_path: Path) -> tuple[dict[str, str], dict[str, float]]:

    cbg_of  = {}
    tinf_of = {}
    with open(tree_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "," not in line:
                continue
            left, right = line.rsplit(",", 1)
            node_id = left.split(".")[-1]
            cbg_of[node_id]  = node_id.rsplit("-", 1)[0]
            tinf_of[node_id] = float(right)
    return cbg_of, tinf_of


def true_distribution(true_cbg_of: dict[str, str],
                      cbg_list: list[str],
                      true_tinf_of: dict[str, float] | None = None,
                      t_cutoff: float | None = None) -> tuple[np.ndarray, int]:

    cbg_index = {c: i for i, c in enumerate(cbg_list)}
    counts = np.zeros(len(cbg_list))
    for node_id, cbg in true_cbg_of.items():
        if t_cutoff is not None and true_tinf_of is not None:
            if true_tinf_of.get(node_id, float("inf")) > t_cutoff:
                continue
        if cbg in cbg_index:
            counts[cbg_index[cbg]] += 1
    total = counts.sum()
    return (counts / total if total > 0 else counts), int(total)


def obs_distribution(nodes: dict, cbg_list: list[str]) -> np.ndarray:
    cbg_index = {c: i for i, c in enumerate(cbg_list)}
    counts = np.zeros(len(cbg_list))
    for node in nodes.values():
        if node["type"] == "observed" and node["cbg"] in cbg_index:
            counts[cbg_index[node["cbg"]]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def recon_distribution(marginals: dict[str, np.ndarray],
                       n_cbg: int) -> np.ndarray:
    
    agg = np.zeros(n_cbg)
    for marg in marginals.values():
        agg += marg
    total = agg.sum()
    return agg / total if total > 0 else agg


def load_population_lookup(path: Path) -> dict[str, int]:
   
    lookup: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or len(row) < 3:
                continue
            geo_id = row[0].strip().strip('"')
            if not geo_id.startswith("1500000US"):
                continue   # skips both header rows
            cbg = geo_id.replace("1500000US", "")
            pop_str = row[2].strip().strip('"')
            try:
                pop = int(pop_str)
            except ValueError:
                continue   # skip rows with missing/non-numeric population
            lookup[cbg] = pop
    return lookup


def population_distribution(cbg_list: list[str],
                            pop_lookup: dict[str, int]) -> tuple[np.ndarray, int, np.ndarray]:
    
    counts  = np.array([pop_lookup.get(cbg, 0) for cbg in cbg_list], dtype=float)
    coverage_mask = np.array([cbg in pop_lookup for cbg in cbg_list], dtype=bool)
    missing = int((~coverage_mask).sum())
    total   = counts.sum()
    return (counts / total if total > 0 else counts), missing, coverage_mask


def restrict_and_renormalize(dist: np.ndarray, mask: np.ndarray) -> np.ndarray:
    
    restricted = dist * mask
    s = restricted.sum()
    return restricted / s if s > 0 else restricted


# ── Core ──────────────────────────────────────────────────────────────────────

def process_simulation(sim_dir: Path,
                       mobility_path: Path,
                       latent_prior_mode: str,
                       label: str,
                       bp_mod,
                       tree_file: str | None = None,
                       pop_lookup: dict[str, int] | None = None) -> bool:

    input_dir = sim_dir / (f"outbreaker_input_{label}" if label else "outbreaker_input")
    tree_path = sim_dir / "tree"
    json_path = input_dir / (tree_file if tree_file else "full_tree.json")

    for p in (json_path, tree_path):
        if not p.exists():
            print(f"  [ERROR] {sim_dir.name}: missing {p.name}")
            return False

    try:
        with open(json_path) as fh:
            tree_data = json.load(fh)

        nodes = {n["node_id"]: n for n in tree_data["nodes"]}
        edges = tree_data["edges"]

        
        true_cbg_of, true_tinf_of = parse_true_tree(tree_path)
        sampled_cbgs = {n["cbg"] for n in nodes.values() if n["cbg"] is not None}
        true_cbgs    = set(true_cbg_of.values())
        cbg_list     = sorted(sampled_cbgs | true_cbgs)
        cbg_index    = {cbg: i for i, cbg in enumerate(cbg_list)}
        n_cbg        = len(cbg_list)

        n_obs = sum(1 for n in nodes.values() if n["type"] == "observed")
        n_lat = sum(1 for n in nodes.values() if n["type"] == "latent")

        print(f"\n  {sim_dir.name}")
        print(f"    Nodes            : {len(nodes)}  ({n_obs} observed, {n_lat} latent)")
        print(f"    CBG state space  : {n_cbg}  "
              f"(sampled={len(sampled_cbgs)}  true-only={len(true_cbgs - sampled_cbgs)})")
        print(f"    True tree nodes  : {len(true_cbg_of)}")

        
        print(f"    Building Q matrix...")
        Q     = bp_mod.build_q_matrix(mobility_path, cbg_list)
        cache = bp_mod.TransitionCache(Q)

        
        if latent_prior_mode == "empirical":
            emp = np.zeros(n_cbg)
            for node in nodes.values():
                if node["type"] == "observed" and node["cbg"] in cbg_index:
                    emp[cbg_index[node["cbg"]]] += 1
            s = emp.sum()
            latent_prior = emp / s if s > 0 else None
            print(f"    Latent prior     : empirical")
        elif latent_prior_mode == "mobility":
            latent_prior = bp_mod.compute_mobility_prior(mobility_path, cbg_list)
            print(f"    Latent prior     : mobility")
        else:
            latent_prior = None
            print(f"    Latent prior     : uniform")

        
        print(f"    Running BP...")
        marginals = bp_mod.run_belief_propagation(
            nodes, edges, cbg_index, cache, latent_prior
        )

        
        obs_tinfs = [n["t_inf"] for n in nodes.values() if n["type"] == "observed"]
        t_cutoff  = max(obs_tinfs) if obs_tinfs else None

        
        dist_true, n_true_used = true_distribution(
            true_cbg_of, cbg_list, true_tinf_of, t_cutoff
        )
        dist_obs   = obs_distribution(nodes, cbg_list)
        dist_recon = recon_distribution(marginals, n_cbg)

        
        if pop_lookup is not None:
            dist_pop, n_pop_missing, pop_coverage_mask = population_distribution(cbg_list, pop_lookup)
            pop_coverage_rate = 1 - (n_pop_missing / n_cbg) if n_cbg > 0 else 0.0
            if n_pop_missing > 0:
                print(f"    [WARN] {n_pop_missing}/{n_cbg} CBGs missing population data "
                      f"(coverage={pop_coverage_rate:.1%})")
            if dist_pop.sum() == 0:
                print(f"    [WARN] No population data matched any CBG in this tree "
                      f"-- skipping population baseline (check CBG ID format/coverage)")
                dist_pop = None
        else:
            dist_pop, n_pop_missing, pop_coverage_mask, pop_coverage_rate = None, None, None, None

        
        h_true  = entropy(dist_true)
        h_obs   = entropy(dist_obs)
        h_recon = entropy(dist_recon)

        err_recon  = abs(h_true - h_recon)
        err_obs    = abs(h_true - h_obs)
        bp_wins    = err_recon < err_obs

        
        ce_obs         = cross_entropy(dist_true, dist_obs)
        ce_recon       = cross_entropy(dist_true, dist_recon)
        ce_wins        = ce_recon < ce_obs
        ce_improvement = ce_obs - ce_recon
        
        ce_ratio       = ce_recon / ce_obs if ce_obs > 0 else float("nan")

        
        n_cbgs_obs   = int((dist_obs   > 0).sum())   # CBGs with observed cases
        n_cbgs_true  = int((dist_true  > 0).sum())   # CBGs with true cases
        n_cbgs_recon = int((dist_recon > 0.01).sum()) # CBGs with >1% recon mass

        
        cost_matrix    = build_cost_matrix(cbg_list)
        emd_obs        = earth_mover_distance(dist_true, dist_obs,   cost_matrix)
        emd_recon      = earth_mover_distance(dist_true, dist_recon, cost_matrix)
        emd_wins       = (emd_recon < emd_obs) if emd_obs is not None else None
        emd_improvement= (emd_obs - emd_recon) if emd_wins is not None else None
        emd_ratio      = (emd_recon / emd_obs) if (emd_obs is not None and emd_obs > 0) else None

        # ── Population baseline metrics (only if dist_pop available) ──────────
        if dist_pop is not None:
            h_pop = entropy(dist_pop)

            
            dist_true_pop_matched = restrict_and_renormalize(dist_true, pop_coverage_mask)
            ce_pop   = cross_entropy(dist_true_pop_matched, dist_pop)
            emd_pop  = earth_mover_distance(dist_true_pop_matched, dist_pop, cost_matrix)
            dist_recon_pop_matched = restrict_and_renormalize(dist_recon, pop_coverage_mask)
            dist_obs_pop_matched   = restrict_and_renormalize(dist_obs,   pop_coverage_mask)
            ce_recon_pop_matched   = cross_entropy(dist_true_pop_matched, dist_recon_pop_matched)
            ce_obs_pop_matched     = cross_entropy(dist_true_pop_matched, dist_obs_pop_matched)

            recon_beats_pop_ce  = ce_recon_pop_matched < ce_pop
            recon_beats_pop_emd = (emd_recon < emd_pop) if (emd_recon is not None and emd_pop is not None) else None
            obs_beats_pop_ce    = ce_obs_pop_matched   < ce_pop
        else:
            h_pop = ce_pop = emd_pop = None
            recon_beats_pop_ce = recon_beats_pop_emd = obs_beats_pop_ce = None

        print(f"\n    t_cutoff         : {t_cutoff:.1f} days")
        print(f"    True nodes used  : {n_true_used} (t_inf <= {t_cutoff:.1f}d)")
        print(f"    H_true           : {h_true:.4f} nats")
        print(f"    H_obs            : {h_obs:.4f} nats  (err={err_obs:.4f})")
        print(f"    H_recon          : {h_recon:.4f} nats  (err={err_recon:.4f})")
        print(f"    BP wins (H)      : {'[Y]' if bp_wins else '[N]'}  "
              f"improvement={err_obs - err_recon:+.4f}")
        print(f"    CBG state space  : {n_cbg} total  "
              f"(obs={n_cbgs_obs}  true={n_cbgs_true}  recon>{'1%'}={n_cbgs_recon})")
        print(f"    CE(true,obs)     : {ce_obs:.4f}  CE(true,recon): {ce_recon:.4f}  "
              f"ratio={ce_ratio:.4f}")
        print(f"    BP wins (CE)     : {'[Y]' if ce_wins else '[N]'}  "
              f"improvement={ce_improvement:+.4f}  ratio={ce_ratio:.4f}")
        if emd_obs is not None:
            print(f"    EMD(true,obs)    : {emd_obs:.2f} km  "
                  f"EMD(true,recon): {emd_recon:.2f} km  ratio={emd_ratio:.4f}")
            print(f"    BP wins (EMD)    : {'[Y]' if emd_wins else '[N]'}  "
                  f"improvement={emd_improvement:+.2f} km")
        if h_pop is not None:
            print(f"\n    H_pop            : {h_pop:.4f} nats  (population baseline)")
            print(f"    Pop coverage     : {pop_coverage_rate:.1%} "
                  f"({n_cbg - n_pop_missing}/{n_cbg} CBGs matched, comparisons coverage-restricted)")
            print(f"    CE(true,pop)     : {ce_pop:.4f}")
            print(f"    Recon beats pop (CE)  : {'[Y]' if recon_beats_pop_ce else '[N]'}")
            if emd_pop is not None:
                print(f"    EMD(true,pop)    : {emd_pop:.2f} km")
                print(f"    Recon beats pop (EMD) : {'[Y]' if recon_beats_pop_emd else '[N]'}")
            print(f"    Obs beats pop (CE)    : {'[Y]' if obs_beats_pop_ce else '[N]'}  "
                  f"(sanity check -- expect [N] if pop baseline is strong)")

       
        out_suffix = f"_{tree_file.replace('.json', '')}" if tree_file else ""

        with open(input_dir / f"entropy_results_empirical{out_suffix}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "sim_name", "label", "latent_prior",
                "n_obs", "n_lat", "n_true", "n_true_used", "t_cutoff", "n_cbgs",
                "n_cbgs_obs", "n_cbgs_true", "n_cbgs_recon",
                "h_true", "h_obs", "h_recon", "h_pop",
                "err_obs", "err_recon", "h_improvement", "bp_wins_h",
                "ce_obs", "ce_recon", "ce_pop", "ce_ratio", "ce_improvement", "bp_wins_ce",
                "emd_obs", "emd_recon", "emd_pop", "emd_ratio", "emd_improvement", "bp_wins_emd",
                "recon_beats_pop_ce", "recon_beats_pop_emd", "obs_beats_pop_ce",
                "pop_coverage_rate", "n_pop_missing",
            ])
            writer.writeheader()
            writer.writerow({
                "sim_name":      sim_dir.name,
                "label":         label or "default",
                "latent_prior":  latent_prior_mode,
                "n_obs":         n_obs,
                "n_lat":         n_lat,
                "n_true":        len(true_cbg_of),
                "n_true_used":   n_true_used,
                "t_cutoff":      round(t_cutoff, 2) if t_cutoff else "",
                "n_cbgs":        n_cbg,
                "n_cbgs_obs":    n_cbgs_obs,
                "n_cbgs_true":   n_cbgs_true,
                "n_cbgs_recon":  n_cbgs_recon,
                "h_true":        round(h_true,  6),
                "h_obs":         round(h_obs,   6),
                "h_recon":       round(h_recon, 6),
                "err_obs":       round(err_obs,        6),
                "err_recon":     round(err_recon,      6),
                "h_improvement": round(err_obs - err_recon, 6),
                "bp_wins_h":     int(bp_wins),
                "ce_obs":        round(ce_obs,          6),
                "ce_recon":      round(ce_recon,        6),
                "ce_ratio":      round(ce_ratio,        6),
                "ce_improvement":round(ce_improvement,  6),
                "bp_wins_ce":    int(ce_wins),
                "emd_obs":       round(emd_obs,         4) if emd_obs         is not None else "",
                "emd_recon":     round(emd_recon,       4) if emd_recon       is not None else "",
                "emd_ratio":     round(emd_ratio,       4) if emd_ratio       is not None else "",
                "emd_improvement":round(emd_improvement,4) if emd_improvement is not None else "",
                "bp_wins_emd":   int(emd_wins)             if emd_wins        is not None else "",
                "h_pop":         round(h_pop,  6) if h_pop  is not None else "",
                "ce_pop":        round(ce_pop, 6) if ce_pop is not None else "",
                "emd_pop":       round(emd_pop, 4) if emd_pop is not None else "",
                "recon_beats_pop_ce":  int(recon_beats_pop_ce)  if recon_beats_pop_ce  is not None else "",
                "recon_beats_pop_emd": int(recon_beats_pop_emd) if recon_beats_pop_emd is not None else "",
                "obs_beats_pop_ce":    int(obs_beats_pop_ce)    if obs_beats_pop_ce    is not None else "",
                "pop_coverage_rate":   round(pop_coverage_rate, 4) if pop_coverage_rate is not None else "",
                "n_pop_missing":       n_pop_missing if n_pop_missing is not None else "",
            })

        # entropy_summary.txt
        with open(input_dir / f"entropy_summary{out_suffix}.txt", "w") as fh:
            fh.write(f"Simulation     : {sim_dir.name}\n")
            fh.write(f"Label          : {label or 'default'}\n")
            fh.write(f"Latent prior   : {latent_prior_mode}\n")
            fh.write(f"Nodes          : {len(nodes)} ({n_obs} obs, {n_lat} latent)\n")
            fh.write(f"True nodes     : {len(true_cbg_of)}\n")
            fh.write(f"True nodes used: {n_true_used} (t_inf <= {t_cutoff:.1f}d)\n")
            fh.write(f"t_cutoff       : {t_cutoff:.1f} days\n")
            fh.write(f"CBG state space: {n_cbg}\n")
            fh.write(f"CBGs observed  : {n_cbgs_obs}\n")
            fh.write(f"CBGs true      : {n_cbgs_true}\n")
            fh.write(f"CBGs recon>1%  : {n_cbgs_recon}\n")
            fh.write(f"\n")
            fh.write(f"H_true         : {h_true:.6f} nats\n")
            fh.write(f"H_obs          : {h_obs:.6f} nats\n")
            fh.write(f"H_recon        : {h_recon:.6f} nats\n")
            fh.write(f"\n")
            fh.write(f"err_obs        : {err_obs:.6f}\n")
            fh.write(f"err_recon      : {err_recon:.6f}\n")
            fh.write(f"h_improvement  : {err_obs - err_recon:+.6f}\n")
            fh.write(f"bp_wins (H)    : {'yes' if bp_wins else 'no'}\n")
            fh.write(f"\n")
            fh.write(f"CE(true,obs)   : {ce_obs:.6f} nats\n")
            fh.write(f"CE(true,recon) : {ce_recon:.6f} nats\n")
            fh.write(f"CE ratio       : {ce_ratio:.6f}\n")
            fh.write(f"CE improvement : {ce_improvement:+.6f}\n")
            fh.write(f"bp_wins (CE)   : {'yes' if ce_wins else 'no'}\n")
            if emd_obs is not None:
                fh.write(f"\n")
                fh.write(f"EMD(true,obs)  : {emd_obs:.4f} km\n")
                fh.write(f"EMD(true,recon): {emd_recon:.4f} km\n")
                fh.write(f"EMD ratio      : {emd_ratio:.4f}\n")
                fh.write(f"EMD improvement: {emd_improvement:+.4f} km\n")
                fh.write(f"bp_wins (EMD)  : {'yes' if emd_wins else 'no'}\n")

        # entropy_plot.png — bar chart of three entropies
        fig, ax = plt.subplots(figsize=(7, 5))
        labels  = ["H_true\n(ground truth)", "H_obs\n(biased sample)", "H_recon\n(BP)"]
        values  = [h_true, h_obs, h_recon]
        colors  = ["#2d6a4f", "#e63946", "#4C72B0"]
        bars    = ax.bar(labels, values, color=colors, width=0.5,
                         edgecolor="white", linewidth=1.5)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Error annotations
        ax.annotate("", xy=(1, h_true), xytext=(1, h_obs),
                    arrowprops=dict(arrowstyle="<->", color="#e63946", lw=1.5))
        ax.text(1.28, (h_true + h_obs) / 2,
                f"err={err_obs:.3f}", color="#e63946", fontsize=9, va="center")

        ax.annotate("", xy=(2, h_true), xytext=(2, h_recon),
                    arrowprops=dict(arrowstyle="<->", color="#4C72B0", lw=1.5))
        ax.text(2.28, (h_true + h_recon) / 2,
                f"err={err_recon:.3f}", color="#4C72B0", fontsize=9, va="center")

        ax.axhline(h_true, color="#2d6a4f", linewidth=1,
                   linestyle="--", alpha=0.5, label="H_true reference")

        ax.set_ylabel("Shannon Entropy (nats)")
        ax.set_title(
            f"{sim_dir.name}\nprior={latent_prior_mode}  "
            f"{'BP better [Y]' if bp_wins else 'Obs better [N]'}  "
            f"improvement={err_obs - err_recon:+.3f}",
            fontweight="bold", fontsize=10
        )
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(input_dir / f"entropy_plot{out_suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"    Written          : entropy_results{out_suffix}.csv, entropy_summary{out_suffix}.txt, entropy_plot{out_suffix}.png")
        return True

    except Exception as e:
        import traceback
        print(f"  [ERROR] {sim_dir.name}: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Single-pass entropy evaluation of BP reconstruction"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Single simulation directory")
    group.add_argument("--all", type=Path, help="Parent directory of all sim dirs")
    parser.add_argument("--mobility",     type=Path, required=True,
                        help="Path to cbg2cbg.csv")
    parser.add_argument("--latent-prior", type=str,  default="empirical",
                        choices=["uniform", "empirical", "mobility"],
                        help="Prior for latent nodes (default: uniform)")
    parser.add_argument("--label",        type=str,  default="",
                        help="Input folder suffix (e.g. '10pct')")
    parser.add_argument("--scripts",      type=Path, default=None,
                        help="Directory containing run_bp.py "
                             "(default: same dir as this script)")
    parser.add_argument("--tree-file",    type=str,  default=None,
                        help="Alternative tree JSON filename in outbreaker_input/ "
                             "(e.g. full_tree_30d.json). Default: full_tree.json")
    parser.add_argument("--population-csv", type=Path, default=None,
                        help="CBG population CSV (Census ACS block-group export, "
                             "e.g. cbgpop.csv with columns GEO_ID,NAME,B01003_001E,B01003_001M). "
                             "Adds population baseline comparison if provided.")
    args = parser.parse_args()

    scripts_dir = args.scripts or Path(__file__).parent
    bp_mod      = load_run_bp(scripts_dir)

    if not args.mobility.exists():
        sys.exit(f"[ERROR] Mobility file not found: {args.mobility}")

    pop_lookup = None
    if args.population_csv:
        if not args.population_csv.exists():
            sys.exit(f"[ERROR] Population CSV not found: {args.population_csv}")
        pop_lookup = load_population_lookup(args.population_csv)
        print(f"Loaded population data for {len(pop_lookup)} CBGs\n")

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
        print(f"Found {len(dirs)} simulation directories\n")

    success = 0
    for sim_dir in dirs:
        ok = process_simulation(sim_dir, args.mobility, args.latent_prior,
                                args.label, bp_mod, args.tree_file, pop_lookup)
        if ok:
            success += 1

    print(f"\nDone: {success}/{len(dirs)} directories processed successfully.")


if __name__ == "__main__":
    main()