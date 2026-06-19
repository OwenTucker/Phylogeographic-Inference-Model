import argparse
import random
import sys
from pathlib import Path
def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id = None
    parts: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(parts)
                current_id = line[1:]   
                parts = []
            else:
                parts.append(line)

    if current_id is not None:
        records[current_id] = "".join(parts)
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")

def parse_tree_dates(path: Path) -> dict[str, int]:
    dates: dict[str, int] = {}
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                print(f"  [WARN] tree line {lineno} has no comma, skipping: {line!r}")
                continue
            left, right = line.rsplit(",", 1)
            node_id = left.split(".")[-1]   # l
            try:
                date = int(right)
            except ValueError:
                print(f"  [WARN] tree line {lineno} non-integer date, skipping: {line!r}")
                continue
            dates[node_id] = date
    return dates
def downsample(cases: list[str], spec: str, rng: random.Random) -> list[str]:
    n = len(cases)
    try:
        value = float(spec)
    except ValueError:
        raise ValueError(f"--downsample must be a number, got: {spec!r}")

    if 0 < value < 1:
        k = max(1, round(n * value))
    elif value >= 1:
        k = min(n, int(value))
    else:
        raise ValueError(f"--downsample must be > 0, got: {value}")

    chosen = set(rng.sample(cases, k))
    return [c for c in cases if c in chosen]
def subsample_by_obs(cases: list[str],
                     tree_dates: dict[str, int],
                     n_obs: int) -> list[str]:

    def sort_key(node_id: str) -> int:
        return tree_dates.get(node_id, 999999)

    sorted_cases = sorted(cases, key=sort_key)
    return sorted_cases[:n_obs]

def process_simulation(sim_dir: Path, downsample_spec: str | None, rng: random.Random, label: str = "", n_obs: int | None = None) -> bool:
    selected_case_path = sim_dir / "selected_case"
    tree_path = sim_dir / "tree"
    fasta_path  = sim_dir / "sampled.fasta"
    out_dir = sim_dir / (f"outbreaker_input_{label}" if label else "outbreaker_input")
    missing = [p for p in (selected_case_path, tree_path, fasta_path) if not p.exists()]
    if missing:
        print(f"  [ERROR] Missing files in {sim_dir.name}: {[p.name for p in missing]}")
        return False
    with open(selected_case_path, "r") as fh:
        selected = [line.strip() for line in fh if line.strip()]

    fasta      = read_fasta(fasta_path)
    tree_dates = parse_tree_dates(tree_path)
    true_tree_size = len(tree_dates)
    n_selected_orig = len(selected)
    sampling_rate = n_selected_orig / true_tree_size if true_tree_size else 0
    print(f"\n  {sim_dir.name}")
    print(f"    True tree size   : {true_tree_size} nodes")
    print(f"    Sampled cases    : {n_selected_orig}  ({sampling_rate:.1%} of true tree)")
    if n_obs is not None:
        if n_obs > len(selected):
            print(f"  [WARN] n_obs={n_obs} > available cases ({len(selected)}), using all")
        else:
            selected = subsample_by_obs(selected, tree_dates, n_obs)
            print(f"    After n_obs cut  : {len(selected)}  "
                  f"(first {n_obs} by infection time, "
                  f"t_max={max(tree_dates.get(s,0) for s in selected)}d)")
    elif downsample_spec is not None:
        selected = downsample(selected, downsample_spec, rng)
        print(f"    After downsample : {len(selected)}  ({len(selected)/true_tree_size:.1%} of true tree)")

    missing_seq   = [s for s in selected if s not in fasta]
    missing_dates = [s for s in selected if s not in tree_dates]

    if missing_seq:
        print(f"  [WARN] {len(missing_seq)} selected cases have no FASTA sequence:")
        for s in missing_seq[:5]:
            print(f"         {s}")
        if len(missing_seq) > 5:
            print(f"         ... and {len(missing_seq) - 5} more")

    if missing_dates:
        print(f"  [WARN] {len(missing_dates)} selected cases have no tree date:")
        for s in missing_dates[:5]:
            print(f"         {s}")
        if len(missing_dates) > 5:
            print(f"         and {len(missing_dates) - 5} more")

    
    out_dir.mkdir(exist_ok=True)
    fasta_records = [
        (node_id, fasta[node_id])
        for node_id in selected
        if node_id in fasta
    ]
    write_fasta(fasta_records, out_dir / "subset.fasta")
    with open(out_dir / "dates.txt", "w") as fh:
        for node_id in selected:
            if node_id in tree_dates:
                fh.write(f"{tree_dates[node_id] + 1}\n")

    n = len(fasta_records)
    out_name = f"outbreaker_input_{label}" if label else "outbreaker_input"
    print(f"    Written          : {n} cases  ->  {out_name}/")
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess simulation dirs for outbreaker2")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path,
                       help="Path to a single simulation directory")
    group.add_argument("--all", type=Path,
                       help="Path to parent directory containing all simulation dirs")
    parser.add_argument("--downsample", type=str, default=None, metavar="N",
                        help="Randomly subsample selected cases: float (0,1] for fraction, "
                             "int >= 1 for absolute count  (e.g. 0.5 or 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible downsampling (default: 42)")
    parser.add_argument("--label", type=str, default="",
                        help="Suffix for output folder name, e.g. '10pct' -> outbreaker_input_10pct/")
    parser.add_argument("--n-obs", type=int, default=None,
                        help="Keep only the first N observed cases by infection time "
                             "(faithful early-outbreak simulation, overrides --downsample)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.dir:
        dirs = [args.dir]
    else:
        parent = args.all
        if not parent.is_dir():
            sys.exit(f"[ERROR] Not a directory: {parent}")
        dirs = sorted(
            d for d in parent.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        print(f"Found {len(dirs)} simulation directories under {parent.name}/")

    if args.downsample:
        print(f"Downsampling with spec={args.downsample!r}, seed={args.seed}")

    success = 0
    for sim_dir in dirs:
        if not sim_dir.is_dir():
            print(f"  [SKIP] Not a directory: {sim_dir}")
            continue
        ok = process_simulation(sim_dir, args.downsample, rng, args.label, args.n_obs)
        if ok:
            success += 1

    print(f"\nDone: {success}/{len(dirs)} directories processed successfully.")


if __name__ == "__main__":
    main()