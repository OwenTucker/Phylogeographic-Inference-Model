import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


def interpolate_times(t_parent: float, t_child: float,
                      kappa: int,
                      gamma_shape: float, gamma_scale: float,
                      rng: random.Random) -> list[float]:
    
    if kappa == 0:
        return []

    total = t_child - t_parent
    if total <= 0:
        # Zero-length or inverted branch — place all latents at t_parent
        return [t_parent] * kappa

    n_gaps = kappa + 1
    # Draw n_gaps gamma values
    raw = [rng.gammavariate(gamma_shape, gamma_scale) for _ in range(n_gaps)]
    raw_sum = sum(raw)
    gaps = [g / raw_sum * total for g in raw]
    times = []
    t = t_parent
    for g in gaps[:-1]:   # last gap lands on t_child (already known)
        t += g
        times.append(round(t, 6))

    return sorted(times)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def read_csv_as_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def extract_cbg(original_id: str) -> str:
    """
    Extract the CBG identifier from an original_id like '360810478001-0'.
    The CBG is everything before the '-' suffix (the case index).
    e.g. '360810478001-0' -> '360810478001'
    """
    return original_id.rsplit("-", 1)[0]


# ── Core builder ──────────────────────────────────────────────────────────────

def build_tree(input_dir: Path,
               gamma_shape: float,
               gamma_scale: float,
               rng: random.Random) -> tuple[dict, list]:
    """
    Build the full annotated tree.

    Returns
    -------
    nodes : dict[node_id -> node_dict]
        Each node has:
          node_id    : str  (unique, e.g. "obs_3" or "lat_3_1")
          type       : "observed" | "latent"
          simple_id  : int | None   (outbreaker 1-indexed id, observed only)
          original_id: str | None   (CBG string, observed only)
          cbg        : str | None   (extracted CBG, observed only; None for latent)
          t_inf      : float        (infection time in days)
          parent     : str | None   (node_id of parent, None for root)
          children   : list[str]    (node_ids of children)

    edges : list[dict]
        Flat table: parent_id, child_id, branch_length, kappa,
                    parent_type, child_type
    """
    # ── load CSVs ─────────────────────────────────────────────────────────────
    map_tree   = read_csv_as_dicts(input_dir / "map_tree.csv")
    map_kappa  = read_csv_as_dicts(input_dir / "map_kappa.csv")
    map_tinf   = read_csv_as_dicts(input_dir / "map_tinf.csv")
    id_mapping = read_csv_as_dicts(input_dir / "id_mapping.csv")

    def _int_or_none(v: str) -> "int | None":
        return None if v.strip() in ("", "NA", "na", "NaN") else int(float(v))

    def _float_or_none(v: str) -> "float | None":
        return None if v.strip() in ("", "NA", "na", "NaN") else float(v)

    kappa_by_child = {
        int(r["child"]): int(float(r["kappa"]))
        for r in map_kappa
        if _int_or_none(r["kappa"]) is not None
    }
    # t_inf should always be valid  skip any NA rows defensively
    tinf_by_id = {
        int(r["child"]): float(r["t_inf"])
        for r in map_tinf
        if _float_or_none(r["t_inf"]) is not None
    }
    info_by_id = {int(r["simple_id"]): r for r in id_mapping}

    parent_of: dict[int, int | None] = {}
    for r in map_tree:
        child  = int(r["child"])
        parent = None if r["parent"] in ("", "NA", "na") else int(float(r["parent"]))
        parent_of[child] = parent

    all_ids = sorted(parent_of.keys())

    nodes: dict[str, dict] = {}

    def obs_id(simple: int) -> str:
        return f"obs_{simple}"

    for sid in all_ids:
        info = info_by_id[sid]
        orig = info["original_id"]
        nodes[obs_id(sid)] = {
            "node_id":     obs_id(sid),
            "type":        "observed",
            "simple_id":   sid,
            "original_id": orig,
            "cbg":         extract_cbg(orig),
            "t_inf":       tinf_by_id[sid],
            "parent":      None,
            "children":    [],
        }

    edges: list[dict] = []

    for child_sid in all_ids:
        parent_sid = parent_of[child_sid]
        if parent_sid is None:
            continue   # root has no parent edge

        kappa      = kappa_by_child.get(child_sid, 0)
        t_child    = tinf_by_id[child_sid]
        t_parent   = tinf_by_id[parent_sid]
        branch_len = t_child - t_parent

        
        n_latent = max(0, kappa - 1)

        if n_latent == 0:
            # Direct edge — no latent nodes (kappa <= 1)
            p_nid = obs_id(parent_sid)
            c_nid = obs_id(child_sid)
            nodes[p_nid]["children"].append(c_nid)
            nodes[c_nid]["parent"] = p_nid
            edges.append({
                "parent_id":    p_nid,
                "child_id":     c_nid,
                "branch_length": round(branch_len, 6),
                "kappa":        kappa,
                "parent_type":  "observed",
                "child_type":   "observed",
            })
        else:
            # Interpolate n_latent intermediate nodes
            latent_times = interpolate_times(
                t_parent, t_child, n_latent, gamma_shape, gamma_scale, rng
            )

            
            latent_ids = [f"lat_{child_sid}_{k}" for k in range(1, n_latent + 1)]
            for lid, lt in zip(latent_ids, latent_times):
                nodes[lid] = {
                    "node_id":     lid,
                    "type":        "latent",
                    "simple_id":   None,
                    "original_id": None,
                    "cbg":         None,
                    "t_inf":       lt,
                    "parent":      None,
                    "children":    [],
                }

            # Chain: observed_parent -> lat_1 -> lat_2 -> ... -> observed_child
            chain = [obs_id(parent_sid)] + latent_ids + [obs_id(child_sid)]
            chain_times = [t_parent] + latent_times + [t_child]

            for i in range(len(chain) - 1):
                p_nid   = chain[i]
                c_nid   = chain[i + 1]
                seg_len = round(chain_times[i + 1] - chain_times[i], 6)
                p_type  = nodes[p_nid]["type"]
                c_type  = nodes[c_nid]["type"]

                nodes[p_nid]["children"].append(c_nid)
                nodes[c_nid]["parent"] = p_nid

                edges.append({
                    "parent_id":     p_nid,
                    "child_id":      c_nid,
                    "branch_length": seg_len,
                    "kappa":         kappa,   # kappa of the original edge
                    "parent_type":   p_type,
                    "child_type":    c_type,
                })

    return nodes, edges


def print_summary(nodes: dict, edges: list, sim_name: str):
    n_obs    = sum(1 for n in nodes.values() if n["type"] == "observed")
    n_lat    = sum(1 for n in nodes.values() if n["type"] == "latent")
    n_edges  = len(edges)
    t_infs   = [n["t_inf"] for n in nodes.values()]
    bl       = [e["branch_length"] for e in edges]
    n_zero   = sum(1 for b in bl if b <= 0)

    
    max_children = max((len(n["children"]) for n in nodes.values()), default=0)
    n_poly   = sum(1 for n in nodes.values() if len(n["children"]) > 2)

    print(f"\n  {sim_name}")
    print(f"    Observed nodes   : {n_obs}")
    print(f"    Latent nodes     : {n_lat}  (total: {n_obs + n_lat})")
    print(f"    Edges            : {n_edges}")
    print(f"    Time range       : {min(t_infs):.1f} — {max(t_infs):.1f} days")
    print(f"    Branch lengths   : min={min(bl):.4f}  mean={sum(bl)/len(bl):.4f}  max={max(bl):.4f}")
    if n_zero:
        print(f"    [WARN] Zero/negative branch lengths: {n_zero}")
    print(f"    Max children     : {max_children}  (polytomies >2 children: {n_poly})")

    
    cbgs = {n["cbg"] for n in nodes.values() if n["cbg"] is not None}
    print(f"    Unique CBGs      : {len(cbgs)}")


def write_outputs(nodes: dict, edges: list, input_dir: Path):
    
    payload = {
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    with open(input_dir / "full_tree.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    
    if edges:
        fieldnames = list(edges[0].keys())
        with open(input_dir / "full_tree.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(edges)


def process_simulation(sim_dir: Path,
                       gamma_shape: float,
                       gamma_scale: float,
                       rng: random.Random,
                       label: str = "") -> bool:
    input_dir = sim_dir / (f"outbreaker_input_{label}" if label else "outbreaker_input")

    required = ["map_tree.csv", "map_kappa.csv", "map_tinf.csv", "id_mapping.csv"]
    missing  = [f for f in required if not (input_dir / f).exists()]
    if missing:
        print(f"  [ERROR] {sim_dir.name}: missing in outbreaker_input/: {missing}")
        return False

    try:
        nodes, edges = build_tree(input_dir, gamma_shape, gamma_scale, rng)
        write_outputs(nodes, edges, input_dir)
        print_summary(nodes, edges, sim_dir.name)
        print(f"    Written          : full_tree.json, full_tree.csv")
        return True
    except Exception as e:
        print(f"  [ERROR] {sim_dir.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build annotated transmission tree from outbreaker2 MAP outputs")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Single simulation directory")
    group.add_argument("--all", type=Path, help="Parent directory of all simulation dirs")
    parser.add_argument("--seed",        type=int,   default=42,  help="Random seed (default: 42)")
    parser.add_argument("--gamma-shape", type=float, default=4.0, help="Gamma shape for gap sampling (default: 4)")
    parser.add_argument("--gamma-scale", type=float, default=2.0, help="Gamma scale for gap sampling (default: 2)")
    parser.add_argument("--label", type=str, default="",
                        help="Input/output folder suffix, e.g. '10pct' -> outbreaker_input_10pct/")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.dir:
        dirs = [args.dir]
    else:
        parent = args.all
        if not parent.is_dir():
            sys.exit(f"[ERROR] Not a directory: {parent}")
        input_folder = f"outbreaker_input_{args.label}" if args.label else "outbreaker_input"
        dirs = sorted(
            d for d in parent.iterdir()
            if d.is_dir() and (d / input_folder).is_dir()
        )
        print(f"Found {len(dirs)} preprocessed simulation directories\n")

    success = 0
    for sim_dir in dirs:
        ok = process_simulation(sim_dir, args.gamma_shape, args.gamma_scale, rng, args.label)
        if ok:
            success += 1

    print(f"\nDone: {success}/{len(dirs)} directories processed successfully.")


if __name__ == "__main__":
    main()