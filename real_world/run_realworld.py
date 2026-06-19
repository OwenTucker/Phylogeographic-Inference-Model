import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def load_run_bp(scripts_dir: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_bp", scripts_dir / "run_bp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_cmd(cmd: list[str], step: str) -> tuple[bool, str]:
    print(f"\n  -> {step}")
    result = subprocess.run(cmd, text=True, capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        err_lines = combined.strip().splitlines()
        short = "\n      ".join(err_lines[-20:]) if err_lines else "(no output captured)"
        print(f"     [FAILED] exit={result.returncode}")
        print(f"      {short}")
        return False, short[:600]
    return True, ""


def entropy(freq_vec: np.ndarray) -> float:
    p = freq_vec[freq_vec > 0]
    return float(-np.sum(p * np.log(p)))

def setup_input_folder(work_dir: Path, label: str) -> Path:
    input_dir = work_dir / f"outbreaker_input_{label}"
    input_dir.mkdir(exist_ok=True)

    files_to_copy = [
        "map_tree.csv", "map_kappa.csv", "map_tinf.csv",
        "id_mapping.csv", "outbreaker_results.rds",
        "subset.fasta", "dates.txt",
    ]
    import shutil
    for fname in files_to_copy:
        src_f = work_dir / fname
        dst   = input_dir / fname
        if src_f.exists() and not dst.exists():
            shutil.copy2(src_f, dst)

    fips_src = work_dir / "id_mapping.csv"
    fips_dst = input_dir / "fips_mapping.csv"
    if fips_src.exists() and not fips_dst.exists():
        shutil.copy2(fips_src, fips_dst)

    return input_dir

def load_fips_mapping(input_dir: Path) -> dict[str, str]:
    mapping = {}
    
    candidates = [input_dir / "fips_mapping.csv",
                  input_dir.parent / "id_mapping.csv",
                  input_dir / "id_mapping.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return mapping
    print(f"    Loading FIPS from: {csv_path.name} ({csv_path.parent.name}/)")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            strain = row.get("strain") or row.get("original_id", "")
            fips   = row.get("fips", "")
            if strain and fips:
                mapping[strain] = fips
    return mapping


def compute_entropy_summary(input_dir: Path,
                            mobility_path: Path,
                            prior: str,
                            bp_mod) -> dict:
    json_path = input_dir / "full_tree.json"
    if not json_path.exists():
        raise FileNotFoundError(f"full_tree.json not found in {input_dir}")

    with open(json_path) as fh:
        tree_data = json.load(fh)

    nodes = {n["node_id"]: n for n in tree_data["nodes"]}
    edges = tree_data["edges"]

    
    fips_map = load_fips_mapping(input_dir)
    if fips_map:
        n_fixed = 0
        for node in nodes.values():
            if node["type"] == "observed" and node.get("original_id"):
                orig = node["original_id"]
                fips = fips_map.get(orig)
                if fips:
                    node["cbg"] = fips
                    n_fixed += 1
        print(f"    FIPS mapped : {n_fixed} observed nodes")
    else:
        print("    [WARN] id_mapping.csv not found or empty — CBGs may be wrong")

    
    cbg_set   = {n["cbg"] for n in nodes.values() if n["cbg"] is not None}
    cbg_list  = sorted(cbg_set)
    cbg_index = {cbg: i for i, cbg in enumerate(cbg_list)}
    n_cbg     = len(cbg_list)
    n_obs     = sum(1 for n in nodes.values() if n["type"] == "observed")
    n_lat     = sum(1 for n in nodes.values() if n["type"] == "latent")

    print(f"\n    Nodes       : {len(nodes)}  ({n_obs} observed, {n_lat} latent)")
    print(f"    CBG space   : {n_cbg}")

    Q     = bp_mod.build_q_matrix(mobility_path, cbg_list)
    cache = bp_mod.TransitionCache(Q)

    # Latent prior
    if prior == "empirical":
        emp = np.zeros(n_cbg)
        for node in nodes.values():
            if node["type"] == "observed" and node["cbg"] in cbg_index:
                emp[cbg_index[node["cbg"]]] += 1
        s = emp.sum()
        latent_prior = emp / s if s > 0 else None
    elif prior == "mobility":
        latent_prior = bp_mod.compute_mobility_prior(mobility_path, cbg_list)
    else:
        latent_prior = None

    print(f"    Running BP  (prior={prior})...")
    marginals = bp_mod.run_belief_propagation(
        nodes, edges, cbg_index, cache, latent_prior
    )

    # Observed distribution (lab-location biased)
    obs_counts = np.zeros(n_cbg)
    for node in nodes.values():
        if node["type"] == "observed" and node["cbg"] in cbg_index:
            obs_counts[cbg_index[node["cbg"]]] += 1
    obs_dist = obs_counts / obs_counts.sum() if obs_counts.sum() > 0 else obs_counts

    # Reconstructed distribution
    recon_counts = np.zeros(n_cbg)
    for marg in marginals.values():
        recon_counts += marg
    recon_dist = recon_counts / recon_counts.sum() if recon_counts.sum() > 0 else recon_counts

    h_obs   = entropy(obs_dist)
    h_recon = entropy(recon_dist)

    print(f"\n    H_obs   : {h_obs:.4f} nats  (lab-location biased)")
    print(f"    H_recon : {h_recon:.4f} nats  (BP reconstructed)")
    print(f"    Delta H : {h_recon - h_obs:+.4f}  "
          f"({'BP spreads distribution' if h_recon > h_obs else 'BP concentrates distribution'})")

    print(f"\n    Top CBGs by reconstructed probability:")
    top_idx = np.argsort(recon_dist)[::-1][:10]
    print(f"    {'CBG':<15} {'obs_freq':>9} {'recon_freq':>11} {'delta':>8}")
    print(f"    {'-'*46}")
    for idx in top_idx:
        cbg = cbg_list[idx]
        print(f"    {cbg:<15} {obs_dist[idx]:>9.4f} {recon_dist[idx]:>11.4f} "
              f"{recon_dist[idx]-obs_dist[idx]:>+8.4f}")

    
    result = {
        "h_obs":        round(h_obs,   6),
        "h_recon":      round(h_recon, 6),
        "delta_h":      round(h_recon - h_obs, 6),
        "n_obs":        n_obs,
        "n_lat":        n_lat,
        "n_cbgs":       n_cbg,
        "prior":        prior,
    }

    with open(input_dir / "realworld_entropy.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    
    with open(input_dir / "realworld_cbg_dist.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cbg", "obs_freq", "recon_freq", "delta"])
        writer.writeheader()
        for i, cbg in enumerate(cbg_list):
            writer.writerow({
                "cbg":        cbg,
                "obs_freq":   round(float(obs_dist[i]),   6),
                "recon_freq": round(float(recon_dist[i]), 6),
                "delta":      round(float(recon_dist[i] - obs_dist[i]), 6),
            })

    print(f"\n    Written: realworld_entropy.csv, realworld_cbg_dist.csv")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Phylogeographic inference pipeline on real-world GISAID data"
    )
    parser.add_argument("--dir",      type=Path, required=True,
                        help="Output dir from preprocess_realworld.py")
    parser.add_argument("--mobility", type=Path, required=True,
                        help="Path to cbg2cbg.csv")
    parser.add_argument("--scripts",  type=Path, default=None,
                        help="Directory containing pipeline scripts")
    parser.add_argument("--prior",    type=str,  default="uniform",
                        choices=["uniform", "empirical", "mobility"])
    parser.add_argument("--rscript",  type=str,  default="Rscript")
    parser.add_argument("--fix-mu",   action="store_true",
                        help="Fix mutation rate at --init-mu instead of estimating")
    parser.add_argument("--no-seq",   action="store_true",
                        help="Dates-only mode: set mu=1e-10 fixed so dates dominate kappa inference")
    parser.add_argument("--init-mu",  type=float, default=0.000002,
                        help="Mutation rate (subs/site/day) for real-world SARS-CoV-2 data. "
                             "Default 0.000002 is the realistic literature rate "
                             "(~0.0007 subs/site/year). DO NOT use the simulation "
                             "default of 0.0018 here -- it causes outbreaker2 crashes "
                             "and degenerate kappa estimates on real sequence data.")
    parser.add_argument("--iter",     type=int,  default=50000)
    parser.add_argument("--label",    type=str,  default="rw",
                        help="Label for outbreaker_input subfolder (default: rw)")
    args = parser.parse_args()

    work_dir    = args.dir
    scripts_dir = args.scripts or Path(__file__).parent
    label       = args.label

    if not work_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {work_dir}")
    if not args.mobility.exists():
        sys.exit(f"[ERROR] Mobility file not found: {args.mobility}")
    if not (work_dir / "subset.fasta").exists():
        sys.exit(f"[ERROR] subset.fasta not found — run preprocess_realworld.py first")

    bp_mod = load_run_bp(scripts_dir)
    t0     = time.time()

    print(f"\n{'='*60}")
    print(f"  REAL-WORLD PIPELINE")
    print(f"  Dir     : {work_dir.name}")
    print(f"  Label   : {label}")
    print(f"  fix_mu  : {args.fix_mu}")
    print(f"  init_mu : {args.init_mu}")
    print(f"{'='*60}")

    input_dir = setup_input_folder(work_dir, label)
    print(f"\n  Input folder: outbreaker_input_{label}/")

    rds_path = input_dir / "outbreaker_results.rds"
    if rds_path.exists():
        print("\n  -> outbreaker2: SKIP (outbreaker_results.rds exists)")
    else:
        extra_flags = ["--init_mu", str(args.init_mu)]
        if args.fix_mu:  extra_flags += ["--fix_mu", "TRUE"]
        if args.no_seq:  extra_flags += ["--no_seq", "TRUE"]
        ok, err = run_cmd([
            args.rscript,
            str(scripts_dir / "outbreaker_run.R"),
            "--dir",   str(work_dir),
            "--iter",  str(args.iter),
            "--label", label,
        ] + extra_flags, "outbreaker_run.R")
        if not ok:
            sys.exit(f"[ERROR] outbreaker2 failed: {err}")
        # Copy RDS back into input_dir if written to work_dir
        import shutil
        for f in ["outbreaker_results.rds", "map_tree.csv", "map_kappa.csv",
                  "map_tinf.csv", "id_mapping.csv", "traces.pdf", "map_tree.pdf"]:
            src_f = work_dir / f
            if src_f.exists() and not (input_dir / f).exists():
                shutil.copy2(src_f, input_dir / f)

    json_path = input_dir / "full_tree.json"
    if json_path.exists():
        print("  -> build_tree: SKIP (full_tree.json exists)")
    else:
        ok, err = run_cmd([
            sys.executable,
            str(scripts_dir / "build_tree.py"),
            "--dir",   str(work_dir),
            "--label", label,
        ], "build_tree.py")
        if not ok:
            sys.exit(f"[ERROR] build_tree failed: {err}")

    print("\n  -> run_bp + entropy...")
    result = compute_entropy_summary(
        input_dir, args.mobility, args.prior, bp_mod
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed:.0f}s)")
    print(f"  H_obs   : {result['h_obs']:.4f} nats")
    print(f"  H_recon : {result['h_recon']:.4f} nats")
    print(f"  Delta H : {result['delta_h']:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()