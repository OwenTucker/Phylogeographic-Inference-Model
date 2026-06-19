import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


import importlib.util

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def major_lineage(lineage: str) -> str:
    parts = lineage.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 else lineage

def find_qualifying_lineages(meta_path: Path,
                             preproc_mod,
                             date_min: str, date_max: str,
                             min_seqs: int, max_concentration: float) -> list[dict]:
   
    rows = []
    with open(meta_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            d = row.get("date", "")
            if not d or d == "?":
                continue
            if date_min and d < date_min:
                continue
            if date_max and d > date_max:
                continue
            rows.append(row)

    by_lineage = defaultdict(list)
    for row in rows:
        lin = major_lineage(row.get("pangolin_lineage", ""))
        by_lineage[lin].append(row)

    qualifying = []
    for lin, group in by_lineage.items():
        # Map each row's location to FIPS
        fips_counts = Counter()
        n_mapped = 0
        for row in group:
            loc = row.get("location", "")
            result = preproc_mod.map_location(loc)
            if result is not None:
                _, fips = result
                fips_counts[fips] += 1
                n_mapped += 1

        if n_mapped < min_seqs:
            continue

        top_county, top_count = fips_counts.most_common(1)[0]
        top_frac = top_count / n_mapped

        if top_frac > max_concentration:
            continue

        qualifying.append({
            "lineage":    lin,
            "n_mapped":   n_mapped,
            "top_county": top_county,
            "top_frac":   round(top_frac, 4),
            "n_counties": len(fips_counts),
        })

    return qualifying

def run_cmd(cmd: list[str], step: str) -> tuple[bool, str]:
    print(f"\n  -> {step}")
    result = subprocess.run(cmd, text=True, capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        err_lines = combined.strip().splitlines()
        short = "\n      ".join(err_lines[-20:]) if err_lines else "(no output captured)"
        print(f"     [FAILED] exit={result.returncode}\n      {short}")
        return False, short[:600]
    out_lines = (result.stdout or "").strip().splitlines()
    for line in out_lines[-8:]:
        print(f"      {line}")
    return True, ""


def cross_entropy(p_true: np.ndarray, q_pred: np.ndarray, eps: float = 1e-10) -> float:
    
    q_safe = np.clip(q_pred, eps, None)
    mask = p_true > 0
    return float(-np.sum(p_true[mask] * np.log(q_safe[mask])))



def load_nyt_cumulative(nyt_dir: Path, target_date: str, fips_filter: set[str]) -> dict[str, int]:
    year = target_date[:4]
    nyt_path = nyt_dir / f"us-counties-{year}.csv"
    if not nyt_path.exists():
        return {}
    latest: dict[str, tuple[str, int]] = {}
    with open(nyt_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fips = row.get("fips", "")
            if fips not in fips_filter:
                continue
            d = row["date"]
            if d > target_date:
                continue
            cases = int(row["cases"]) if row["cases"] else 0
            if fips not in latest or d > latest[fips][0]:
                latest[fips] = (d, cases)
    return {fips: count for fips, (_, count) in latest.items()}


def get_date_range(lineage_dir: Path) -> tuple[str, str]:
    csv_path = lineage_dir / "id_mapping.csv"
    dates = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dates.append(row["date"])
    return min(dates), max(dates)


GA_COUNTY_POPULATION = {
    "13021": 157346, "13063": 297595, "13121": 1066710, "13089": 764382,
    "13067": 766149, "13135": 957062, "13051": 295291, "13151": 234561,
    "13097": 144720, "13077": 149255, "13153": 156335, "13285": 69298,
    "13113": 119194, "13183": 19798, "13071": 35376, "13073": 156714,
    "13095": 89208, "13173": 19619, "13295": 68756, "13039": 54412,
    "13297": 94593, "13057": 266620, "13313": 105108, "13047": 67843,
    "13059": 128331, "13117": 251283, "13139": 204441, "13045": 119247,
    "13157": 75317, "13245": 200421, "13225": 26554, "13215": 206922,
    "13031": 79608, "13127": 85292, "13305": 28699, "13053": 10907,
    "13315": 9105, "13009": 44890, "13293": 26635, "13189": 21389,
    "13085": 27092, "13123": 28258, "13291": 24364, "13195": 30197,
    "13043": 10996, "13255": 25319, "13275": 44451, "13063": 297595,
}


def evaluate_lineage(lineage_dir: Path, nyt_dir: Path) -> dict | None:
    cbg_dist_path = lineage_dir / "outbreaker_input_rw" / "realworld_cbg_dist.csv"
    if not cbg_dist_path.exists():
        return None

    cbg_rows  = list(csv.DictReader(open(cbg_dist_path, newline="", encoding="utf-8")))
    tree_fips = {r["cbg"] for r in cbg_rows}
    obs_map   = {r["cbg"]: float(r["obs_freq"])   for r in cbg_rows}
    recon_map = {r["cbg"]: float(r["recon_freq"]) for r in cbg_rows}

    date_min, date_max = get_date_range(lineage_dir)
    print(f"    NYT window: {date_min} to {date_max}")

    all_ga = sorted(GA_COUNTY_POPULATION.keys())

    obs_full   = np.array([obs_map.get(f, 0.0)   for f in all_ga])
    recon_full = np.array([recon_map.get(f, 0.0) for f in all_ga])
    obs_full   = obs_full   / obs_full.sum()   if obs_full.sum()   > 0 else obs_full
    recon_full = recon_full / recon_full.sum() if recon_full.sum() > 0 else recon_full
    pop_counts = np.array([GA_COUNTY_POPULATION[f] for f in all_ga], dtype=float)
    pop_full   = pop_counts / pop_counts.sum()

    years = sorted(set([date_min[:4], date_max[:4]]))
    cum_start: dict[str, int] = {}
    cum_end:   dict[str, int] = {}
    for year in years:
        nyt_path = nyt_dir / f"us-counties-{year}.csv"
        if not nyt_path.exists():
            continue
        with open(nyt_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fips = row.get("fips", "")
                if not fips.startswith("13"):
                    continue
                d     = row["date"]
                cases = int(row["cases"]) if row["cases"] else 0
                if d <= date_min:
                    cum_start[fips] = cases
                if d <= date_max:
                    cum_end[fips] = cases

    nyt_counts = np.array([
        max(0, cum_end.get(f, 0) - cum_start.get(f, 0))
        for f in all_ga
    ], dtype=float)
    nyt_total = nyt_counts.sum()
    if nyt_total == 0:
        return None
    nyt_full = nyt_counts / nyt_total

    def _jsd(p, q):
        m = 0.5 * (p + q)
        kl_pm = np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
        kl_qm = np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    jsd_obs   = _jsd(nyt_full, obs_full)
    jsd_recon = _jsd(nyt_full, recon_full)
    jsd_pop   = _jsd(nyt_full, pop_full)

    
    active = nyt_counts > 0
    nyt_a  = nyt_full[active]; nyt_a = nyt_a / nyt_a.sum()

    def _norm(v): s = v.sum(); return v / s if s > 0 else v
    eps = 1e-10
    ce_obs_r   = float(-np.sum(nyt_a * np.log(_norm(obs_full[active])   + eps)))
    ce_recon_r = float(-np.sum(nyt_a * np.log(_norm(recon_full[active]) + eps)))
    ce_pop_r   = float(-np.sum(nyt_a * np.log(_norm(pop_full[active])   + eps)))

    return {
        "date_min":        date_min,
        "date_max":        date_max,
        "n_counties":      len(tree_fips),
        "n_active":        int(active.sum()),
        "nyt_total_cases": int(nyt_total),
        "jsd_obs":         round(jsd_obs,    6),
        "jsd_recon":       round(jsd_recon,  6),
        "jsd_pop":         round(jsd_pop,    6),
        "ce_r_obs":        round(ce_obs_r,   6),
        "ce_r_recon":      round(ce_recon_r, 6),
        "ce_r_pop":        round(ce_pop_r,   6),
        "jsd_beats_obs":   jsd_recon < jsd_obs,
        "jsd_beats_pop":   jsd_recon < jsd_pop,
        "ce_r_beats_obs":  ce_recon_r < ce_obs_r,
        "ce_r_beats_pop":  ce_recon_r < ce_pop_r,
    }

def process_lineage(lineage: str, fasta: Path, meta: Path, out_root: Path,
                    mobility: Path, nyt_dir: Path, scripts_dir: Path,
                    date_min: str, date_max: str,
                    fix_mu: bool, prior: str, rscript: str, iters: int,
                    init_mu: float = 0.000002,
                    n_obs: int | None = None) -> dict:
    lineage_dir = out_root / lineage.replace(".", "_")
    lineage_dir.mkdir(parents=True, exist_ok=True)

    status = {"lineage": lineage, "n_seqs": "", "step": "", "error": ""}

    print(f"\n{'='*60}\n  LINEAGE: {lineage}\n{'='*60}")

    
    fasta_out = lineage_dir / "subset.fasta"
    if not fasta_out.exists():
        cmd = [sys.executable, str(scripts_dir / "preprocess_realworld.py"),
               "--fasta", str(fasta), "--meta", str(meta),
               "--out", str(lineage_dir), "--lineage", lineage,
               "--date-min", date_min, "--date-max", date_max]
        if n_obs is not None:
            cmd += ["--n-obs", str(n_obs)]
        ok, err = run_cmd(cmd, "preprocess_realworld.py")
        if not ok:
            status["error"] = f"preprocess: {err}"
            return status

    if fasta_out.exists():
        with open(fasta_out) as fh:
            status["n_seqs"] = sum(1 for line in fh if line.startswith(">"))

    
    entropy_csv = lineage_dir / "outbreaker_input_rw" / "realworld_entropy.csv"
    if not entropy_csv.exists():
        cmd = [sys.executable, str(scripts_dir / "run_realworld.py"),
               "--dir", str(lineage_dir), "--mobility", str(mobility),
               "--scripts", str(scripts_dir), "--prior", prior,
               "--rscript", rscript, "--iter", str(iters),
               "--init-mu", str(init_mu)]
        if fix_mu:
            cmd.append("--fix-mu")
        ok, err = run_cmd(cmd, "run_realworld.py")
        if not ok:
            status["error"] = f"pipeline: {err}"
            return status

    
    eval_result = evaluate_lineage(lineage_dir, nyt_dir)
    if eval_result is None:
        status["error"] = "evaluation: realworld_cbg_dist.csv missing"
        return status

    status.update(eval_result)
    status["step"] = "complete"
    return status


def main():
    parser = argparse.ArgumentParser(description="Full real-world batch with cross-entropy evaluation")
    parser.add_argument("--fasta",    type=Path, required=True)
    parser.add_argument("--meta",     type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--nyt-dir",  type=Path, required=True)
    parser.add_argument("--scripts",  type=Path, default=None)
    parser.add_argument("--min-seqs", type=int, default=30)
    parser.add_argument("--max-concentration", type=float, default=0.8)
    parser.add_argument("--date-min", type=str, default="2021-01-01")
    parser.add_argument("--date-max", type=str, default="2023-12-31")
    parser.add_argument("--fix-mu",   action="store_true")
    parser.add_argument("--init-mu",  type=float, default=0.000002,
                        help="Mutation rate (subs/site/day), default 0.000002 "
                             "for real SARS-CoV-2 data")
    parser.add_argument("--prior",    type=str, default="uniform",
                        choices=["uniform", "empirical", "mobility"])
    parser.add_argument("--rscript",  type=str, default="Rscript")
    parser.add_argument("--iter",     type=int, default=50000)
    parser.add_argument("--n-obs",    type=int, default=None,
                        help="Subsample to first N sequences by date per lineage "
                             "(default: use all qualifying sequences). Set to 50 "
                             "to replicate the simulation's early-outbreak regime.")
    args = parser.parse_args()

    scripts_dir = args.scripts or Path(__file__).parent
    args.out_root.mkdir(parents=True, exist_ok=True)
    preproc_mod = load_module(scripts_dir / "preprocess_realworld.py", "preprocess_realworld")

    print(f"Identifying qualifying lineages...")
    print(f"  min_seqs={args.min_seqs}  max_concentration={args.max_concentration}")
    qualifying = find_qualifying_lineages(
        args.meta, preproc_mod, args.date_min, args.date_max,
        args.min_seqs, args.max_concentration
    )

    qualifying.sort(key=lambda q: -q["n_mapped"])

    print(f"\nQualifying lineages ({len(qualifying)}):")
    print(f"{'Lineage':<10} {'N':>5} {'Top county':>11} {'Top frac':>9} {'N counties':>11}")
    print("-" * 55)
    for q in qualifying:
        print(f"{q['lineage']:<10} {q['n_mapped']:>5} {q['top_county']:>11} "
              f"{q['top_frac']:>9.2%} {q['n_counties']:>11}")

    if not qualifying:
        sys.exit("[ERROR] No lineages passed the filter.")

    
    all_status = []
    for q in qualifying:
        result = process_lineage(
            lineage=q["lineage"], fasta=args.fasta, meta=args.meta,
            out_root=args.out_root, mobility=args.mobility, nyt_dir=args.nyt_dir,
            scripts_dir=scripts_dir, date_min=args.date_min, date_max=args.date_max,
            fix_mu=args.fix_mu, prior=args.prior, rscript=args.rscript, iters=args.iter,
            init_mu=args.init_mu, n_obs=args.n_obs,
        )
        all_status.append(result)

        summary_path = args.out_root / "batch_summary.csv"
        fieldnames = ["lineage", "n_seqs", "step", "date_min", "date_max",
                     "n_counties", "n_active", "nyt_total_cases",
                     "jsd_obs", "jsd_recon", "jsd_pop",
                     "ce_r_obs", "ce_r_recon", "ce_r_pop",
                     "jsd_beats_obs", "jsd_beats_pop",
                     "ce_r_beats_obs", "ce_r_beats_pop", "error"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_status)

    
    print(f"\n{'='*90}")
    print(f"  SUMMARY  (JSD open-world + CE restricted to active counties)")
    print(f"{'='*90}")
    print(f"  {'Lineage':<10} {'N':>5} {'JSD_obs':>8} {'JSD_rc':>8} {'JSD_pop':>8} "
          f"{'CE_r_obs':>9} {'CE_r_rc':>8} {'CE_r_pop':>9}  {'JSD<obs':>7} {'JSD<pop':>7} {'CE<obs':>6} {'CE<pop':>6}")
    print("  " + "-" * 103)
    for s in all_status:
        if s.get("error"):
            print(f"  {s['lineage']:<10} {str(s['n_seqs']):>5}  FAILED: {s['error'][:50]}")
            continue
        print(f"  {s['lineage']:<10} {s['n_seqs']:>5} "
              f"{s.get('jsd_obs',0):>8.4f} {s.get('jsd_recon',0):>8.4f} {s.get('jsd_pop',0):>8.4f} "
              f"{s.get('ce_r_obs',0):>9.4f} {s.get('ce_r_recon',0):>8.4f} {s.get('ce_r_pop',0):>9.4f}  "
              f"{'YES' if s.get('jsd_beats_obs') else 'no':>7} "
              f"{'YES' if s.get('jsd_beats_pop') else 'no':>7} "
              f"{'YES' if s.get('ce_r_beats_obs') else 'no':>6} "
              f"{'YES' if s.get('ce_r_beats_pop') else 'no':>6}")

    completed = [s for s in all_status if not s.get("error")]
    if completed:
        print(f"\n  JSD  beats obs: {sum(bool(s.get('jsd_beats_obs'))   for s in completed)}/{len(completed)}")
        print(f"  JSD  beats pop: {sum(bool(s.get('jsd_beats_pop'))   for s in completed)}/{len(completed)}")
        print(f"  CE_r beats obs: {sum(bool(s.get('ce_r_beats_obs')) for s in completed)}/{len(completed)}")
        print(f"  CE_r beats pop: {sum(bool(s.get('ce_r_beats_pop')) for s in completed)}/{len(completed)}")

    print(f"\n  Summary saved: {args.out_root / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()