import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


LABEL = None   # set from args at runtime


def input_dir(sim_dir: Path, label: str) -> Path:
    return sim_dir / f"outbreaker_input_{label}"


def skip_step1(sim_dir: Path, label: str) -> bool:
    return (input_dir(sim_dir, label) / "subset.fasta").exists()


def skip_step2(sim_dir: Path, label: str) -> bool:
    return (input_dir(sim_dir, label) / "outbreaker_results.rds").exists()


def skip_step3(sim_dir: Path, label: str) -> bool:
    return (input_dir(sim_dir, label) / "full_tree.json").exists()


def skip_step4(sim_dir: Path, label: str) -> bool:
    return (input_dir(sim_dir, label) / "bp_marginals.csv").exists()


def skip_step5(sim_dir: Path, label: str) -> bool:
    return (input_dir(sim_dir, label) / "entropy_results.csv").exists()


def run_cmd(cmd: list[str], step_name: str,
            stream: bool = False) -> tuple[bool, str]:
    print(f"    -> {step_name}")
    if stream:
        result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"      [FAILED] exit={result.returncode}")
            return False, f"exit {result.returncode}"
        return True, ""
    else:
        result = subprocess.run(cmd, text=True, capture_output=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            err_lines = (result.stderr or result.stdout or "").strip().splitlines()
            short_err = "\n".join(err_lines[-10:])
            print(f"      [FAILED] exit={result.returncode}")
            print(f"      {short_err}")
            return False, short_err[:300]
        return True, ""


def process_tree(sim_dir: Path,
                 mobility_path: Path,
                 n_obs: int,
                 label: str,
                 prior: str,
                 rscript: str,
                 scripts_dir: Path) -> dict:
   
    t0 = time.time()
    status = {
        "sim_name":  sim_dir.name,
        "label":     label,
        "step1_pre": "skip",
        "step2_ob2": "skip",
        "step3_tree":"skip",
        "step4_bp":  "skip",
        "step5_ent": "skip",
        "h_true":        "",
        "h_obs":         "",
        "h_recon":       "",
        "h_improvement": "",
        "ce_improvement":"",
        "emd_improvement":"",
        "bp_wins_h":     "",
        "bp_wins_ce":    "",
        "bp_wins_emd":   "",
        "error":     "",
        "elapsed":   "",
    }

    print(f"\n  {'='*56}")
    print(f"  {sim_dir.name}")
    print(f"  {'='*56}")

   
    if skip_step1(sim_dir, label):
        print(f"    -> preprocess: SKIP (subset.fasta exists)")
    else:
        ok, err = run_cmd([
            sys.executable,
            str(scripts_dir / "preprocess_outbreaker.py"),
            "--dir",   str(sim_dir),
            "--n-obs", str(n_obs),
            "--label", label,
        ], "preprocess_outbreaker.py")
        status["step1_pre"] = "ok" if ok else "FAIL"
        if not ok:
            status["error"] = f"step1: {err}"
            status["elapsed"] = round(time.time() - t0, 1)
            return status
    status["step1_pre"] = status["step1_pre"] if status["step1_pre"] == "FAIL" else "ok"

   
    if skip_step2(sim_dir, label):
        print(f"    -> outbreaker2: SKIP (outbreaker_results.rds exists)")
    else:
        ok, err = run_cmd([
            rscript,
            str(scripts_dir / "outbreaker_run.R"),
            "--dir",   str(sim_dir),
            "--label", label,
        ], "outbreaker_run.R")
        status["step2_ob2"] = "ok" if ok else "FAIL"
        if not ok:
            status["error"] = f"step2: {err}"
            status["elapsed"] = round(time.time() - t0, 1)
            return status
    status["step2_ob2"] = status["step2_ob2"] if status["step2_ob2"] == "FAIL" else "ok"

    
    if skip_step3(sim_dir, label):
        print(f"    -> build_tree: SKIP (full_tree.json exists)")
    else:
        ok, err = run_cmd([
            sys.executable,
            str(scripts_dir / "build_tree.py"),
            "--dir",   str(sim_dir),
            "--label", label,
        ], "build_tree.py")
        status["step3_tree"] = "ok" if ok else "FAIL"
        if not ok:
            status["error"] = f"step3: {err}"
            status["elapsed"] = round(time.time() - t0, 1)
            return status
    status["step3_tree"] = status["step3_tree"] if status["step3_tree"] == "FAIL" else "ok"

    
    if skip_step4(sim_dir, label):
        print(f"    -> run_bp: SKIP (bp_marginals.csv exists)")
    else:
        ok, err = run_cmd([
            sys.executable,
            str(scripts_dir / "run_bp.py"),
            "--dir",          str(sim_dir),
            "--mobility",     str(mobility_path),
            "--latent-prior", prior,
            "--label",        label,
        ], "run_bp.py", stream=True)
        status["step4_bp"] = "ok" if ok else "FAIL"
        if not ok:
            status["error"] = f"step4: {err}"
            status["elapsed"] = round(time.time() - t0, 1)
            return status
    status["step4_bp"] = status["step4_bp"] if status["step4_bp"] == "FAIL" else "ok"

    if skip_step5(sim_dir, label):
        print(f"    -> run_entropy: SKIP (entropy_results.csv exists)")
    else:
        ok, err = run_cmd([
            sys.executable,
            str(scripts_dir / "run_entropy.py"),
            "--dir",          str(sim_dir),
            "--mobility",     str(mobility_path),
            "--latent-prior", prior,
            "--label",        label,
            "--scripts",      str(scripts_dir),
        ], "run_entropy.py", stream=True)
        status["step5_ent"] = "ok" if ok else "FAIL"
        if not ok:
            status["error"] = f"step5: {err}"
            status["elapsed"] = round(time.time() - t0, 1)
            return status
    status["step5_ent"] = status["step5_ent"] if status["step5_ent"] == "FAIL" else "ok"

    
    entropy_csv = input_dir(sim_dir, label) / "entropy_results.csv"
    if entropy_csv.exists():
        with open(entropy_csv, newline="") as fh:
            row = next(csv.DictReader(fh))
        status["h_true"]         = row.get("h_true", "")
        status["h_obs"]          = row.get("h_obs", "")
        status["h_recon"]        = row.get("h_recon", "")
        status["h_improvement"]  = row.get("h_improvement", "")
        status["ce_improvement"] = row.get("ce_improvement", "")
        status["emd_improvement"]= row.get("emd_improvement", "")
        status["bp_wins_h"]      = "[Y]" if row.get("bp_wins_h")   == "1" else "[N]"
        status["bp_wins_ce"]     = "[Y]" if row.get("bp_wins_ce")  == "1" else "[N]"
        status["bp_wins_emd"]    = "[Y]" if row.get("bp_wins_emd") == "1" else "[N]"

        print(f"\n    H_true={status['h_true']}  H_obs={status['h_obs']}  "
              f"H_recon={status['h_recon']}  "
              f"H:{status['h_improvement']} {status['bp_wins_h']}  "
              f"CE:{status['ce_improvement']} {status['bp_wins_ce']}  "
              f"EMD:{status['emd_improvement']} {status['bp_wins_emd']}")

    status["elapsed"] = round(time.time() - t0, 1)
    return status



STATUS_FIELDS = [
    "sim_name", "label",
    "step1_pre", "step2_ob2", "step3_tree", "step4_bp", "step5_ent",
    "h_true", "h_obs", "h_recon", "h_improvement", "ce_improvement", "emd_improvement", "bp_wins_h", "bp_wins_ce", "bp_wins_emd",
    "error", "elapsed"
]


def load_existing_status(status_path: Path) -> dict[str, dict]:
    existing = {}
    if status_path.exists():
        with open(status_path, newline="") as fh:
            for row in csv.DictReader(fh):
                existing[row["sim_name"]] = row
    return existing


def write_status(status_path: Path, all_status: list[dict]):
    with open(status_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(all_status)

def main():
    parser = argparse.ArgumentParser(
        description="Batch pipeline: preprocess -> outbreaker2 -> tree -> BP -> entropy"
    )
    parser.add_argument("--all",      type=Path, required=True,
                        help="Parent directory containing all simulation dirs")
    parser.add_argument("--mobility", type=Path, required=True,
                        help="Path to cbg2cbg.csv")
    parser.add_argument("--n-obs",    type=int,  default=50,
                        help="Observed cases cutoff (default: 50)")
    parser.add_argument("--prior",    type=str,  default="uniform",
                        choices=["uniform", "empirical", "mobility"],
                        help="Latent prior for BP (default: uniform)")
    parser.add_argument("--rscript",  type=str,  default="Rscript",
                        help="Path to Rscript (default: Rscript)")
    parser.add_argument("--scripts",  type=Path, default=None,
                        help="Directory containing pipeline scripts "
                             "(default: same dir as this script)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run all steps even if outputs exist")
    args = parser.parse_args()

    parent      = args.all
    scripts_dir = args.scripts or Path(__file__).parent
    label       = f"{args.n_obs}obs"

    if not parent.is_dir():
        sys.exit(f"[ERROR] Not a directory: {parent}")
    if not args.mobility.exists():
        sys.exit(f"[ERROR] Mobility file not found: {args.mobility}")

    
    sim_dirs = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (d / "tree").exists()
    )
    print(f"Found {len(sim_dirs)} simulation directories")
    print(f"Label: {label}  Prior: {args.prior}  N-obs: {args.n_obs}\n")

    status_path = parent / "batch_status.csv"
    existing    = load_existing_status(status_path)
    if args.force:
        existing = {}
        print("--force: re-running all steps\n")

    all_status: list[dict] = []
    n_complete = 0
    n_failed   = 0
    n_skipped  = 0

    for i, sim_dir in enumerate(sim_dirs):
        name = sim_dir.name

        if name in existing and existing[name].get("step5_ent") == "ok":
            print(f"  [{i+1:02d}/{len(sim_dirs)}] SKIP (complete): {name}")
            all_status.append(existing[name])
            n_skipped += 1
            continue

        print(f"\n  [{i+1:02d}/{len(sim_dirs)}]")
        result = process_tree(
            sim_dir      = sim_dir,
            mobility_path = args.mobility,
            n_obs        = args.n_obs,
            label        = label,
            prior        = args.prior,
            rscript      = args.rscript,
            scripts_dir  = scripts_dir,
        )
        all_status.append(result)
        write_status(status_path, all_status)

        if result["error"]:
            n_failed += 1
            print(f"  [FAILED] {name}")
        else:
            n_complete += 1

    
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  Total    : {len(sim_dirs)}")
    print(f"  Complete : {n_complete + n_skipped}")
    print(f"  Failed   : {n_failed}")
    print(f"{'='*60}\n")

    # Print entropy summary table
    completed = [s for s in all_status if s.get("h_true")]
    if completed:
        wins_h   = sum(1 for s in completed if s.get("bp_wins_h")   == "[Y]")
        wins_ce  = sum(1 for s in completed if s.get("bp_wins_ce")  == "[Y]")
        wins_emd = sum(1 for s in completed if s.get("bp_wins_emd") == "[Y]")
        print(f"  {'Simulation':<55} {'H_true':>7} {'H_obs':>7} "
              f"{'H_recon':>7} {'dH':>7} {'dCE':>7} {'dEMD':>7} {'H/CE/EMD':>9}")
        print(f"  {'-'*105}")
        for s in completed:
            dh   = f"{float(s['h_improvement']):>+7.3f}"  if s.get("h_improvement")   else "      ?"
            dce  = f"{float(s['ce_improvement']):>+7.3f}" if s.get("ce_improvement")  else "      ?"
            demd = f"{float(s['emd_improvement']):>+7.1f}"if s.get("emd_improvement") else "      ?"
            wins = f"{s.get('bp_wins_h','?')}/{s.get('bp_wins_ce','?')}/{s.get('bp_wins_emd','?')}"
            print(f"  {s['sim_name']:<55} "
                  f"{float(s['h_true']):>7.3f} "
                  f"{float(s['h_obs']):>7.3f} "
                  f"{float(s['h_recon']):>7.3f} "
                  f"{dh} {dce} {demd} {wins:>9}")
        print(f"\n  BP wins  H: {wins_h}/{len(completed)}  "
              f"CE: {wins_ce}/{len(completed)}  "
              f"EMD: {wins_emd}/{len(completed)}")

    print(f"\n  Status saved: {status_path}")


if __name__ == "__main__":
    main()