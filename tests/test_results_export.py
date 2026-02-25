from pathlib import Path

import openpyxl

from sph.io.results_export import ResultsLogger, StepMetrics, VxProfileMetrics


def test_results_export_csv_and_xlsx(tmp_path: Path):
    logger = ResultsLogger(meta={"scene_file": "dummy.json", "solver_type": "wcsph"})
    logger.log_step(
        StepMetrics(
            step=1,
            t=0.1,
            dt=0.1,
            vmax=1.0,
            rho_min=990.0,
            rho_avg=1000.0,
            rho_max=1010.0,
            err_avg_pct=0.0,
            p_min=-10.0,
            p_avg=0.0,
            p_max=10.0,
            neigh_min=5,
            neigh_avg=10.0,
            neigh_max=15,
        )
    )
    logger.log_step(
        StepMetrics(
            step=2,
            t=0.2,
            dt=0.1,
            vmax=2.0,
            rho_min=995.0,
            rho_avg=1001.0,
            rho_max=1012.0,
            err_avg_pct=0.1,
            p_min=-5.0,
            p_avg=1.0,
            p_max=9.0,
            neigh_min=6,
            neigh_avg=11.0,
            neigh_max=16,
        )
    )
    logger.log_vxprof(VxProfileMetrics(step=2, bins=4, x_window=0.0, vx_mean=[0.0, 0.1, 0.2, 0.1]))

    logger.export(tmp_path, base_name="run", formats=("csv", "xlsx"))

    steps_csv = tmp_path / "run_steps.csv"
    vx_csv = tmp_path / "run_vxprof.csv"
    xlsx = tmp_path / "run.xlsx"

    assert steps_csv.exists() and steps_csv.stat().st_size > 0
    assert vx_csv.exists() and vx_csv.stat().st_size > 0
    assert xlsx.exists() and xlsx.stat().st_size > 0

    # CSV headers
    header = steps_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "step" in header and "dt" in header and "rho_avg" in header and "neigh_max" in header

    # XLSX sheets
    wb = openpyxl.load_workbook(xlsx)
    assert set(["steps", "vx_profile", "meta"]).issubset(set(wb.sheetnames))


