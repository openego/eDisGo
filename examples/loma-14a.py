from edisgo import EDisGo
from datetime import datetime
import geopandas as gpd


def run_optimization_14a(edisgo):
    """
    Run optimization with §14a curtailment enabled.

    Uses opf_version=5 which uses §14a curtailment as the only flexibility tool.
    Minimizes line losses and §14a usage. Grid restrictions (voltage 0.9-1.1 p.u.,
    current limits) are enforced as hard constraints. Feasibility slacks exist but
    are penalized at 1e8 to ensure the model remains feasible.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with time series

    Returns
    -------
    EDisGo
        EDisGo object with optimization results
    """
    print(f"\n{'='*80}")
    print(f"⚡ Running OPF with §14a Curtailment")
    print(f"{'='*80}")
    print(f"\nUsing OPF version 5:")
    print(f"  - §14a curtailment as only flexibility tool")
    print(f"  - Minimize line losses + §14a usage")
    print(f"  - Grid restrictions enforced (voltage 0.9-1.1, current limits)")
    print(f"  - Feasibility slacks penalized at 1e8")

    start_time = datetime.now()

    # Run optimization
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    return edisgo

grid_path="/home/carlos/LoMa/exec_folder/results/MGB_model_pypsa"
edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(120, 148))
mv_grid_geom = gpd.read_file(
    "/home/carlos/LoMa/exec_folder/data/Input_files/MV_grid_district/husum_district.shp"
)
mv_grid_geom = mv_grid_geom.to_crs(4326)
edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0,"geometry"]
edisgo.topology.check_integrity()
pypsa_n = edisgo.to_pypsa()
edisgo.analyze()
edisgo = run_optimization_14a(edisgo)
