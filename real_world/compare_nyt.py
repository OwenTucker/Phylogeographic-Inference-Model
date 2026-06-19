import argparse
import csv
from pathlib import Path

import numpy as np


GA_COUNTY_POPULATION = {
    "13001": 18910,  "13003": 11975,  "13005": 27606,  "13007": 23700,
    "13009": 44890,  "13011": 17634,  "13013": 107621, "13015": 110727,
    "13017": 18842,  "13019": 16478,  "13021": 157346, "13023": 11120,
    "13025": 17800,  "13027": 44072,  "13029": 35884,  "13031": 79608,
    "13033": 12517,  "13035": 24630,  "13037": 17914,  "13039": 54412,
    "13043": 10996,  "13045": 119247, "13047": 67843,  "13049": 290213,
    "13051": 295291, "13053": 10907,  "13055": 7281,   "13057": 266620,
    "13059": 128331, "13061": 7093,   "13063": 297595, "13065": 57004,
    "13067": 766149, "13069": 22736,  "13071": 35376,  "13073": 156714,
    "13075": 126719, "13077": 149255, "13079": 29966,  "13081": 29744,
    "13083": 30233,  "13085": 27092,  "13087": 14219,  "13089": 764382,
    "13091": 19807,  "13093": 29171,  "13095": 89208,  "13097": 144720,
    "13099": 19189,  "13101": 22404,  "13103": 6165,   "13105": 27744,
    "13107": 5765,   "13109": 22453,  "13111": 118143, "13113": 119194,
    "13115": 263014, "13117": 251283, "13119": 23432,  "13121": 1066710,
    "13123": 28258,  "13125": 8424,   "13127": 85292,  "13129": 25590,
    "13131": 5009,   "13133": 929119, "13135": 957062, "13137": 29555,
    "13139": 204441, "13141": 19896,  "13143": 65017,  "13145": 38964,
    "13147": 10217,  "13149": 57051,  "13151": 234561, "13153": 156335,
    "13155": 20734,  "13157": 75317,  "13159": 10760,  "13161": 8820,
    "13163": 45989,  "13165": 12093,  "13167": 35167,  "13169": 16177,
    "13171": 18113,  "13173": 19619,  "13175": 39929,  "13177": 14052,
    "13179": 57010,  "13181": 10080,  "13183": 10096,  "13185": 16491,
    "13187": 17413,  "13189": 21389,  "13191": 26329,  "13193": 25967,
    "13195": 30197,  "13197": 9358,   "13199": 17767,  "13201": 15193,
    "13205": 20166,  "13207": 7988,   "13209": 44871,  "13211": 36522,
    "13213": 112432, "13215": 206922, "13217": 9463,   "13219": 164635,
    "13221": 28776,  "13223": 168706, "13225": 26554,  "13227": 66369,
    "13229": 19123,  "13231": 23450,  "13233": 13966,  "13235": 17006,
    "13237": 14884,  "13239": 27302,  "13241": 202518, "13243": 692952,
    "13245": 200421, "13247": 8519,   "13249": 16677,  "13251": 71432,
    "13253": 30461,  "13255": 25319,  "13257": 26715,  "13259": 6829,
    "13261": 18204,  "13263": 32808,  "13265": 23980,  "13267": 9697,
    "13269": 11253,  "13271": 44612,  "13273": 25510,  "13275": 44451,
    "13277": 6847,   "13279": 7979,   "13281": 32606,  "13283": 12212,
    "13285": 69298,  "13287": 32923,  "13289": 70535,  "13291": 24364,
    "13293": 26635,  "13295": 68756,  "13297": 94593,  "13299": 25689,
    "13301": 15259,  "13303": 10523,  "13305": 28699,  "13307": 16439,
    "13309": 27532,  "13311": 10998,  "13313": 105108, "13315": 9105,
    "13317": 88255,  "13319": 9537,   "13321": 7870,
}

ALL_GA_FIPS = sorted(GA_COUNTY_POPULATION.keys())

def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
   
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask] + eps)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
    kl_qm = np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def get_date_range(lineage_dir: Path) -> tuple[str, str]:
    csv_path = lineage_dir / "id_mapping.csv"
    dates = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dates.append(row["date"])
    return min(dates), max(dates)


def load_nyt_all_ga(nyt_dir: Path, date_min: str, date_max: str) -> dict[str, int]:
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
    return {
        f: max(0, cum_end.get(f, 0) - cum_start.get(f, 0))
        for f in ALL_GA_FIPS
    }


def main():
    parser = argparse.ArgumentParser(
        description="JSD + restricted-CE comparison against NYT cases"
    )
    parser.add_argument("--lineage-dir", type=Path, required=True)
    parser.add_argument("--nyt-dir",     type=Path, required=True)
    args = parser.parse_args()

    cbg_dist_path = args.lineage_dir / "outbreaker_input_rw" / "realworld_cbg_dist.csv"
    if not cbg_dist_path.exists():
        raise FileNotFoundError(f"realworld_cbg_dist.csv not found: {cbg_dist_path}")

    cbg_rows  = list(csv.DictReader(open(cbg_dist_path, newline="", encoding="utf-8")))
    tree_fips = {r["cbg"] for r in cbg_rows}
    obs_map   = {r["cbg"]: float(r["obs_freq"])   for r in cbg_rows}
    recon_map = {r["cbg"]: float(r["recon_freq"]) for r in cbg_rows}

    date_min, date_max = get_date_range(args.lineage_dir)
    print(f"Lineage dir  : {args.lineage_dir.name}")
    print(f"Tree counties: {len(tree_fips)}")
    print(f"Date window  : {date_min} to {date_max}")

    obs_full   = np.array([obs_map.get(f, 0.0)   for f in ALL_GA_FIPS])
    recon_full = np.array([recon_map.get(f, 0.0) for f in ALL_GA_FIPS])
    obs_full   = obs_full   / obs_full.sum()   if obs_full.sum()   > 0 else obs_full
    recon_full = recon_full / recon_full.sum() if recon_full.sum() > 0 else recon_full

    pop_counts = np.array([GA_COUNTY_POPULATION[f] for f in ALL_GA_FIPS], dtype=float)
    pop_full   = pop_counts / pop_counts.sum()

    nyt_incremental = load_nyt_all_ga(args.nyt_dir, date_min, date_max)
    nyt_counts = np.array([nyt_incremental.get(f, 0) for f in ALL_GA_FIPS], dtype=float)
    nyt_total  = nyt_counts.sum()
    if nyt_total == 0:
        print("[WARN] No NYT cases found in window.")
        return
    nyt_full = nyt_counts / nyt_total
    print(f"NYT total    : {int(nyt_total)} cases across {(nyt_counts > 0).sum()} counties\n")

    jsd_obs   = jsd(nyt_full, obs_full)
    jsd_recon = jsd(nyt_full, recon_full)
    jsd_pop   = jsd(nyt_full, pop_full)

    active_mask  = nyt_counts > 0
    n_active     = int(active_mask.sum())
    nyt_active   = nyt_full[active_mask]
    nyt_active  /= nyt_active.sum()

    obs_active   = obs_full[active_mask];   s = obs_active.sum();   obs_active   = obs_active/s   if s > 0 else obs_active
    recon_active = recon_full[active_mask]; s = recon_active.sum(); recon_active = recon_active/s if s > 0 else recon_active
    pop_active   = pop_full[active_mask];   s = pop_active.sum();   pop_active   = pop_active/s   if s > 0 else pop_active

    ce_obs_r   = cross_entropy(nyt_active, obs_active)
    ce_recon_r = cross_entropy(nyt_active, recon_active)
    ce_pop_r   = cross_entropy(nyt_active, pop_active)

    print(f"{'='*65}")
    print(f"  METRIC 1: JSD (open-world, all {len(ALL_GA_FIPS)} GA counties)")
    print(f"  Bounded [0, ln2={np.log(2):.4f}], lower = better match to NYT")
    print(f"{'='*65}")
    print(f"  JSD(nyt, obs)   : {jsd_obs:.4f}")
    print(f"  JSD(nyt, recon) : {jsd_recon:.4f}")
    print(f"  JSD(nyt, pop)   : {jsd_pop:.4f}")
    print(f"  Recon beats obs : {'YES' if jsd_recon < jsd_obs else 'NO'}")
    print(f"  Recon beats pop : {'YES' if jsd_recon < jsd_pop else 'NO'}")

    print(f"\n{'='*65}")
    print(f"  METRIC 2: CE restricted to NYT-active counties (n={n_active})")
    print(f"  Mirrors simulation evaluation: only counties with true cases")
    print(f"{'='*65}")
    print(f"  CE(nyt, obs)   : {ce_obs_r:.4f}")
    print(f"  CE(nyt, recon) : {ce_recon_r:.4f}")
    print(f"  CE(nyt, pop)   : {ce_pop_r:.4f}")
    print(f"  Recon beats obs : {'YES' if ce_recon_r < ce_obs_r else 'NO'}")
    print(f"  Recon beats pop : {'YES' if ce_recon_r < ce_pop_r else 'NO'}")

    out_csv = args.lineage_dir / "nyt_comparison_final.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "fips", "nyt_freq", "obs_freq", "recon_freq", "pop_freq",
            "nyt_cases", "in_tree", "nyt_active"
        ])
        writer.writeheader()
        for i, f in enumerate(ALL_GA_FIPS):
            writer.writerow({
                "fips":       f,
                "nyt_freq":   round(float(nyt_full[i]),   6),
                "obs_freq":   round(float(obs_full[i]),   6),
                "recon_freq": round(float(recon_full[i]), 6),
                "pop_freq":   round(float(pop_full[i]),   6),
                "nyt_cases":  int(nyt_counts[i]),
                "in_tree":    int(f in tree_fips),
                "nyt_active": int(nyt_counts[i] > 0),
            })

    summary = args.lineage_dir / "nyt_comparison_final_summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Lineage dir   : {args.lineage_dir.name}\n")
        fh.write(f"Date window   : {date_min} to {date_max}\n")
        fh.write(f"NYT total     : {int(nyt_total)} cases\n")
        fh.write(f"Active counties: {n_active}\n\n")
        fh.write(f"JSD (open-world, all {len(ALL_GA_FIPS)} GA counties):\n")
        fh.write(f"  JSD(nyt, obs)   : {jsd_obs:.6f}\n")
        fh.write(f"  JSD(nyt, recon) : {jsd_recon:.6f}\n")
        fh.write(f"  JSD(nyt, pop)   : {jsd_pop:.6f}\n")
        fh.write(f"  Recon beats obs : {jsd_recon < jsd_obs}\n")
        fh.write(f"  Recon beats pop : {jsd_recon < jsd_pop}\n\n")
        fh.write(f"CE restricted to NYT-active counties (n={n_active}):\n")
        fh.write(f"  CE(nyt, obs)   : {ce_obs_r:.6f}\n")
        fh.write(f"  CE(nyt, recon) : {ce_recon_r:.6f}\n")
        fh.write(f"  CE(nyt, pop)   : {ce_pop_r:.6f}\n")
        fh.write(f"  Recon beats obs : {ce_recon_r < ce_obs_r}\n")
        fh.write(f"  Recon beats pop : {ce_recon_r < ce_pop_r}\n")

    print(f"\n  Written: {out_csv.name}, {summary.name}")


if __name__ == "__main__":
    main()