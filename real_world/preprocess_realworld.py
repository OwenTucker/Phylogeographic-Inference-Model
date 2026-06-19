import argparse
import csv
import random
import sys
from datetime import date, datetime
from pathlib import Path

LOCATION_TO_FIPS = {
    "bibb county":            ("Bibb County",         "13021"),
    "clayton county":         ("Clayton County",      "13063"),
    "fulton county":          ("Fulton County",        "13121"),
    "dekalb county":          ("DeKalb County",        "13089"),
    "effingham county":       ("Effingham County",     "13103"),
    "peach county":           ("Peach County",         "13225"),
    "chatham county":         ("Chatham County",       "13051"),
    "forsyth county":         ("Forsyth County",       "13117"),
    "colquitt county":        ("Colquitt County",      "13071"),
    "chattahoochee county":   ("Chattahoochee County", "13053"),
    "cobb county":            ("Cobb County",          "13067"),
    "dougherty county":       ("Dougherty County",     "13095"),
    "fayette county":         ("Fayette County",       "13113"),
    "hall county":            ("Hall County",          "13139"),
    "jackson county":         ("Jackson County",       "13157"),
    "thomas county":          ("Thomas County",        "13275"),
    "wilkinson county":       ("Wilkinson County",     "13315"),
    "bulloch":                ("Bulloch County",       "13031"),
    "cobb":                   ("Cobb County",          "13067"),
    "columbia":               ("Columbia County",      "13073"),
    "dekalb":                 ("DeKalb County",        "13089"),
    "fayette":                ("Fayette County",       "13113"),
    "fulton":                 ("Fulton County",        "13121"),
    "gwinnett":               ("Gwinnett County",      "13135"),
    "houston":                ("Houston County",       "13153"),
    "liberty":                ("Liberty County",       "13179"),
    "troup":                  ("Troup County",         "13285"),
    "atlanta":                ("Fulton County",        "13121"),
    "alpharetta":             ("Fulton County",        "13121"),
    "roswell":                ("Fulton County",        "13121"),
    "sandy springs":          ("Fulton County",        "13121"),
    "east point":             ("Fulton County",        "13121"),
    "union city":             ("Fulton County",        "13121"),
    "johns creek":            ("Fulton County",        "13121"),
    "milton":                 ("Fulton County",        "13121"),
    "college park":           ("Fulton County",        "13121"),
    "marietta":               ("Cobb County",          "13067"),
    "smyrna":                 ("Cobb County",          "13067"),
    "kennesaw":               ("Cobb County",          "13067"),
    "mableton":               ("Cobb County",          "13067"),
    "austell":                ("Cobb County",          "13067"),
    "powder springs":         ("Cobb County",          "13067"),
    "acworth":                ("Cobb County",          "13067"),
    "decatur":                ("DeKalb County",        "13089"),
    "brookhaven":             ("DeKalb County",        "13089"),
    "tucker":                 ("DeKalb County",        "13089"),
    "stone mountain":         ("DeKalb County",        "13089"),
    "lithonia":               ("DeKalb County",        "13089"),
    "dunwoody":               ("DeKalb County",        "13089"),
    "doraville":              ("DeKalb County",        "13089"),
    "ellenwood":              ("Henry County",         "13151"),  # border area
    "jonesboro":              ("Clayton County",       "13063"),
    "riverdale":              ("Clayton County",       "13063"),
    "hampton":                ("Henry County",         "13151"),
    "mcdonough":              ("Henry County",         "13151"),
    "stockbridge":            ("Henry County",         "13151"),
    "stonecrest":             ("DeKalb County",        "13089"),
    "lawrenceville":          ("Gwinnett County",      "13135"),
    "duluth":                 ("Gwinnett County",      "13135"),
    "norcross":               ("Gwinnett County",      "13135"),
    "snellville":             ("Gwinnett County",      "13135"),
    "loganville":             ("Walton County",        "13297"),
    "buford":                 ("Gwinnett County",      "13135"),
    "savannah":               ("Chatham County",       "13051"),
    "garden city":            ("Chatham County",       "13051"),
    "rincon":                 ("Effingham County",     "13103"),
    "springfield":            ("Effingham County",     "13103"),
    "macon":                  ("Bibb County",          "13021"),
    "warner robins":          ("Houston County",       "13153"),
    "columbus":               ("Muscogee County",      "13215"),
    "augusta":                ("Richmond County",      "13245"),
    "martinez":               ("Columbia County",      "13073"),
    "evans":                  ("Columbia County",      "13073"),
    "albany":                 ("Dougherty County",     "13095"),
    "athens":                 ("Clarke County",        "13059"),
    "commerce":               ("Jackson County",       "13157"),
    "cumming":                ("Forsyth County",       "13117"),
    "sugar hill":             ("Gwinnett County",      "13135"),
    "flowery branch":         ("Hall County",          "13139"),
    "gainesville":            ("Hall County",          "13139"),
    "canton":                 ("Cherokee County",      "13057"),
    "woodstock":              ("Cherokee County",      "13057"),
    "dalton":                 ("Whitfield County",     "13313"),
    "cohutta":                ("Whitfield County",     "13313"),
    "ringgold":               ("Catoosa County",       "13047"),
    "rossville":              ("Walker County",        "13295"),
    "lookout mountain":       ("Walker County",        "13295"),
    "flintstone":             ("Walker County",        "13295"),
    "carrollton":             ("Carroll County",       "13045"),
    "villa rica":             ("Carroll County",       "13045"),
    "whitesburg":             ("Carroll County",       "13045"),
    "newnan":                 ("Coweta County",        "13077"),
    "senoia":                 ("Coweta County",        "13077"),
    "sharpsburg":             ("Coweta County",        "13077"),
    "douglasville":           ("Douglas County",       "13097"),
    "hiram":                  ("Paulding County",      "13223"),
    "dallas":                 ("Paulding County",      "13223"),
    "peachtree city":         ("Fayette County",       "13113"),
    "fayetteville":           ("Fayette County",       "13113"),
    "griffin":                ("Spalding County",      "13255"),
    "thomasville":            ("Thomas County",        "13275"),
    "milledgeville":          ("Baldwin County",       "13009"),
    "thomaston":              ("Upson County",         "13293"),
    "lagrange":               ("Troup County",         "13285"),
    "statesboro":             ("Bulloch County",       "13031"),
    "metter":                 ("Candler County",       "13043"),
    "brunswick":              ("Glynn County",         "13127"),
    "kingsland":              ("Camden County",        "13039"),
    "saint marys":            ("Camden County",        "13039"),
    "woodbine":               ("Camden County",        "13039"),
    "jesup":                  ("Wayne County",         "13305"),
    "ludowici":               ("Long County",          "13183"),
    "richmond hill":          ("Bryan County",         "13029"),
    "toccoa":                 ("Stephens County",      "13257"),
    "blairsville":            ("Union County",         "13291"),
    "ellijay":                ("Gilmer County",        "13123"),
    "dawsonville":            ("Dawson County",        "13085"),
    "colbert":                ("Madison County",       "13195"),
    "thomson":                ("McDuffie County",      "13189"),
    "lithia springs":         ("Douglas County",       "13097"),
    "lakeland":               ("Lanier County",        "13173"),
    "peachtree corners":      ("Gwinnett County",      "13135"),
    "cartersvilke":           ("Bartow County",        "13015"),  # Cartersville typo
    "claytoncounty":          ("Clayton County",       "13063"),
    "savannav":               ("Chatham County",       "13051"),  # Savannah typo
}


def normalize_location(loc: str) -> str:
    if not loc or not isinstance(loc, str):
        return ""
    return " ".join(loc.lower().strip().split())

def map_location(loc: str) -> tuple[str, str] | None:
    norm = normalize_location(loc)
    if not norm:
        return None

    if norm in LOCATION_TO_FIPS:
        return LOCATION_TO_FIPS[norm]

    for suffix in [" county", " co.", " co"]:
        if norm.endswith(suffix):
            base = norm[:-len(suffix)].strip()
            if base + " county" in LOCATION_TO_FIPS:
                return LOCATION_TO_FIPS[base + " county"]

    return None
def read_fasta(path: Path) -> dict[str, str]:
    records = {}
    current_id = None
    parts = []
    with open(path) as fh:
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


def write_fasta(records: list[tuple[str, str]], path: Path):
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")


def parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def date_to_int(d: date, origin: date) -> int:
    return (d - origin).days + 1   
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess real-world GISAID sequences for phylogeographic inference"
    )
    parser.add_argument("--fasta",  type=Path, required=True,
                        help="Aligned FASTA file")
    parser.add_argument("--meta",   type=Path, required=True,
                        help="Metadata TSV file")
    parser.add_argument("--out",    type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--n-obs",  type=int,  default=None,
                        help="Subsample to first N sequences by date")
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--date-min", type=str, default=None,
                        help="Keep sequences on or after this date (YYYY-MM-DD)")
    parser.add_argument("--date-max", type=str, default=None,
                        help="Keep sequences on or before this date (YYYY-MM-DD)")
    parser.add_argument("--lineage",  type=str, default=None,
                        help="Filter to only this pangolin_lineage value "
                             "(e.g. 'BA.1'). Exact match on the metadata column.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.lineage:
        print(f"Filtering to lineage: {args.lineage}")
    print("Loading metadata...")
    meta_rows = []
    with open(args.meta, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            meta_rows.append(row)
    print(f"  {len(meta_rows)} sequences in metadata")

  
    print("Loading FASTA...")
    fasta = read_fasta(args.fasta)
    print(f"  {len(fasta)} sequences in FASTA")

    
    print("Filtering and mapping locations...")
    valid    = []
    unmapped = []

    for row in meta_rows:
        strain   = row["strain"]
        loc      = row.get("location", "")
        date_str = row.get("date", "")
        lineage  = row.get("pangolin_lineage", "")

        
        if args.lineage is not None:
            parts = lineage.split(".")
            major = ".".join(parts[:2]) if len(parts) > 2 else lineage
            if major != args.lineage:
                continue

        
        if strain not in fasta:
            continue

        
        d = parse_date(date_str)
        if d is None:
            continue

        
        fips_result = map_location(loc)
        if fips_result is None:
            unmapped.append(loc)
            continue

        county_name, fips = fips_result
        valid.append({
            "strain":      strain,
            "date":        d,
            "location":    loc,
            "county":      county_name,
            "fips":        fips,
        })

    print(f"  Mapped:   {len(valid)}")
    print(f"  Unmapped: {len(unmapped)} sequences")

    
    unmapped_counts = {}
    for loc in unmapped:
        unmapped_counts[loc] = unmapped_counts.get(loc, 0) + 1
    with open(args.out / "unmapped.txt", "w") as fh:
        for loc, count in sorted(unmapped_counts.items(),
                                  key=lambda x: -x[1]):
            fh.write(f"{count:4d}  {loc}\n")

    if not valid:
        sys.exit("[ERROR] No valid sequences after filtering")

    valid.sort(key=lambda r: r["date"])

    date_min = parse_date(args.date_min) if args.date_min else None
    date_max = parse_date(args.date_max) if args.date_max else None
    if date_min or date_max:
        before = len(valid)
        valid = [
            r for r in valid
            if (date_min is None or r["date"] >= date_min)
            and (date_max is None or r["date"] <= date_max)
        ]
        dmin_str = args.date_min or "any"
        dmax_str = args.date_max or "any"
        print(f"  Date filter {dmin_str} -> {dmax_str}: {before} -> {len(valid)} sequences")
        valid.sort(key=lambda r: r["date"])

    if args.n_obs is not None and args.n_obs < len(valid):
        selected = valid[:args.n_obs]
        print(f"  Subsampled to first {args.n_obs} by date "
              f"(date range: {selected[0]['date']} to {selected[-1]['date']})")
    else:
        selected = valid
        print(f"  Using all {len(selected)} valid sequences "
              f"(date range: {selected[0]['date']} to {selected[-1]['date']})")

    origin = selected[0]["date"]
    for row in selected:
        row["date_int"] = date_to_int(row["date"], origin)

    fasta_records = [(r["strain"], fasta[r["strain"]]) for r in selected]
    write_fasta(fasta_records, args.out / "subset.fasta")

    with open(args.out / "dates.txt", "w") as fh:
        for r in selected:
            fh.write(f"{r['date_int']}\n")
    with open(args.out / "id_mapping.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "simple_id", "strain", "fips", "county", "location",
            "date", "date_int"
        ])
        writer.writeheader()
        for i, r in enumerate(selected, 1):
            writer.writerow({
                "simple_id": i,
                "strain":    r["strain"],
                "fips":      r["fips"],
                "county":    r["county"],
                "location":  r["location"],
                "date":      r["date"].isoformat(),
                "date_int":  r["date_int"],
            })

    
    fips_counts = {}
    for r in selected:
        fips_counts[r["fips"]] = fips_counts.get(r["fips"], 0) + 1

    print(f"\n  Output directory : {args.out}")
    print(f"  subset.fasta     : {len(selected)} sequences")
    print(f"  dates.txt        : range {selected[0]['date_int']} to "
          f"{selected[-1]['date_int']} days")
    print(f"  Unique counties  : {len(fips_counts)}")
    print(f"\n  Top counties:")
    for fips, count in sorted(fips_counts.items(),
                               key=lambda x: -x[1])[:10]:
        county = next(r["county"] for r in selected if r["fips"] == fips)
        print(f"    {county:<25} {fips}  n={count}")


if __name__ == "__main__":
    main()