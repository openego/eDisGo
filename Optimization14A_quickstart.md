# §14a EnWG Curtailment Analysis - Quick Start Guide

## What does `analyze_14a.py` do?

This script performs a complete §14a curtailment analysis for heat pumps and charging points in a distribution grid:

1. **Loads a grid** (ding0 format)
2. **Adds heat pumps** (50 units, 11-20 kW, realistic size distribution)
3. **Adds charging points** (30 units, 3.7-50 kW, home/work/fast charging)
4. **Generates realistic winter timeseries** (configurable days, default: 7 days)
5. **Runs OPF with §14a curtailment** (opf_version=3)
6. **Analyzes curtailment results** (statistics, daily/monthly aggregation)
7. **Creates visualizations** (3 plots: timeseries, HP profiles, CP profiles)
8. **Exports CSV data** (summary, curtailment data, totals per HP/CP)

**Output:** Results folder with plots and CSV files for detailed analysis.

---

## Setup & Run (Complete Workflow)

### Create Project Directory

```bash
# Create and navigate to project directory
mkdir -p ~/projects/edisgo_14a
cd ~/projects/edisgo_14a
```

### Clone eDisGo Repository

```bash
# Clone the repository
git clone https://github.com/openego/eDisGo.git
cd eDisGo

# Checkout the §14a feature branch
git checkout project/411_LoMa_14aOptimization_with_virtual_generators
```

### Setup Python Environment

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install eDisGo

```bash
# Install eDisGo in development mode
python -m pip install -e .[dev]  # install eDisGo from source
```

The script uses grid data from the `30879` folder:

```bash
# Navigate to workspace root
cd ~/projects/edisgo_14a

# Grid folder should be at:
# ~/projects/edisgo_14a/30879/
# with files: buses.csv, lines.csv, generators.csv, etc.

# Verify grid data exists
ls -lh 30879/
```

### Run the Analysis

```bash
# Navigate to script location
cd ~/projects/edisgo_14a

# Run the analysis
python analyze_14a.py
```

**Expected runtime:** 5-15 minutes (depending on hardware)

---

## Configuration

Edit these variables in `analyze_14a.py` (around line 1015):

```python
# Grid configuration
GRID_PATH = "./30879"          # Path to ding0 grid folder
SCENARIO = "eGon2035"          # Scenario name

# Simulation parameters
NUM_DAYS = 7                   # Number of days to simulate (7, 30, 365)
NUM_HEAT_PUMPS = 50           # Number of heat pumps to add
NUM_CHARGING_POINTS = 30      # Number of charging points to add

# Output
OUTPUT_DIR = "./"             # Where to save results
```

---

## Output Files

After running, you'll find a results folder:

```
results_7d_HP50_CP30_14a/
├── summary_statistics.csv          # Overall statistics
├── curtailment_timeseries.csv      # Hourly curtailment per HP/CP
├── curtailment_daily.csv           # Daily aggregation
├── curtailment_monthly.csv         # Monthly aggregation
├── hp_curtailment_total.csv        # Total curtailment per HP
├── cp_curtailment_total.csv        # Total curtailment per CP
├── curtailment_timeseries.png      # Time series plot
├── detailed_hp_profiles.png        # Detailed HP profile
└── detailed_cp_profiles.png        # Detailed CP profile (if curtailed)
```

---

## Using Your Own Grid

### Option 1: Use Another ding0 Grid

```python
# In analyze_14a.py, change GRID_PATH:
GRID_PATH = "/path/to/your/ding0/grid/folder"

# Grid folder must contain:
# - buses.csv
# - lines.csv
# - generators.csv
# - loads.csv
# - transformers.csv
# - etc.
```
