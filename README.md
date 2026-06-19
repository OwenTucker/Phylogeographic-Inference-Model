# Phylogeographic Inference via Mobility-Informed Belief Propagation

We propose a pipeline that recovers the geographic distribution of an early-stage infectious disease outbreak from income-biased genomic surveillance data. Given a small set of sequenced cases, we infer a transmission tree via outbreaker2, augment it with unsampled intermediate cases using kappa-based interpolation, and run CTMC belief propagation parameterized by Advan human mobility flows to produce a reconstructed geographic case distribution. We evaluate against ground-truth simulated outbreak trees and real-world GISAID/NYT COVID-19 data from Georgia, USA.


## Repository Structure

```
├── preprocess\_outbreaker.py       

├── outbreaker\_run.R                

├── build\_tree.py                  
├── run\_bp.py                      
├── run\_entropy.py                 

├── run\_batch.py                   

&#x09;real\_world/
	├── preprocess\_realworld.py        
	├── run\_realworld.py              
	├── run\_realworld\_batch.py         

&#x09;├── compare\_nyt.py                 
```


## Dependencies

### Python

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

### R

```r
install.packages("outbreaker2")
install.packages("ape")
install.packages("igraph")
```

Tested with R 4.4+. outbreaker2 version 1.1.0+.

\---

## External Data

\---

The following external datasets are required and are **not included** in this repository due to licensing restrictions. Obtain them separately and place them as described below.GISAID data requires a GISAID account and adherence to the GISAID Data Access Agreement. NYT data is available at: https://github.com/nytimes/covid-19-data

\---

## Reproducing Simulation Results (Table 1)

The simulation pipeline runs on 40 synthetic outbreak trees across R0 = {1.25, 1.3, 1.4, 1.5}.

### (preprocess → outbreaker2 → build tree → BP → entropy)

Run once per latent prior. Each run writes `entropy\_results.csv` (uniform),
`entropy\_results\_empirical.csv`, and `entropy\_results\_mobility.csv` to each tree's
`outbreaker\_input\_50obs/` directory.

```bash
# Uniform prior (default)
python run\_batch.py \\
    --all path/to/biased\_sampled\_cbg \\
    --mobility path/to/cbg2cbg.csv \\
    --population-csv path/to/cbgpop\_nationwide.csv \\
    --n-obs 50 --prior uniform --label 50obs

# Empirical prior
python run\_entropy.py \\
    --all path/to/biased\_sampled\_cbg \\
    --mobility path/to/cbg2cbg.csv \\
    --population-csv path/to/cbgpop\_nationwide.csv \\
    --latent-prior empirical --label 50obs

# Mobility prior
python run\_entropy.py \\
    --all path/to/biased\_sampled\_cbg \\
    --mobility path/to/cbg2cbg.csv \\
    --population-csv path/to/cbgpop\_nationwide.csv \\
    --latent-prior mobility --label 50obs
```

Note: `run\_batch.py` runs outbreaker2 and BP for the uniform prior end-to-end.
The empirical and mobility entropy runs reuse the same outbreaker/tree outputs —
only the BP and evaluation steps re-run (skipping the expensive outbreaker step).

### Step 2: Generate Table 1

```bash
python aggregate\_prior\_comparison.py \\
    --all path/to/biased\_sampled\_cbg \\
    --label 50obs \\
    --out-csv results/prior\_r0\_summary.csv \\
    --out-tex results/table1.tex
```

## Reproducing Real-World Results

The real-world pipeline runs on GISAID SARS-CoV-2 sequences from Georgia, split by major
Pango lineage. Lineages with fewer than 30 sequences or more than 80% of cases in a single
county are excluded.

### Step 1: Run the batch pipeline across all qualifying lineages

```bash
cd path/to/real\_world/

python run\_realworld\_batch.py \\
    --fasta aligned.fasta \\
    --meta metadata.tsv \\
    --out-root real\_world\_batch \\
    --mobility ../county2county.csv \\
    --nyt-dir NYT\_data \\
    --scripts path/to/scripts \\
    --fix-mu --prior uniform \\
    --min-seqs 30 --max-concentration 0.8
```

This will:

1. Identify qualifying lineages (BA.1, BA.5, BA.2, BA.4, BQ.1, AY.39, BF.10)
2. Preprocess each lineage's FASTA and metadata
3. Run outbreaker2 with the corrected SARS-CoV-2 mutation rate (2×10⁻⁶ subs/site/day)
4. Build the augmented transmission tree and run BP
5. Evaluate against NYT time-matched case counts using JSD (open-world) and
CE restricted to NYT-active counties
6. Write `batch\_summary.csv` with all results

### Step 2: Per-lineage comparison (optional, for detailed per-county tables)

```bash
python compare\_nyt.py \\
    --lineage-dir real\_world\_batch/BA\_4 \\
    --nyt-dir NYT\_data
```

\---

## 

