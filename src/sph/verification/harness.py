from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.core.vx_profile import compute_vx_profile
from sph.neighbors.spatial_hash import SpatialHash
from sph.verification.metrics import (
    density_error_stats,
    has_non_finite_state,
    kinetic_energy_total,
    l2_error,
    linf_error,
    mass_total,
    momentum_total,
)


def _as_gate(*, value: float | bool, pass_flag: bool, max_value: float | None = None, min_value: float | None = None) -> dict:
    return {
        "value": value,
        "min": min_value,
        "max": max_value,
        "pass": bool(pass_flag),
    }


def _build_cfg(scene: dict) -> SimConfig:
    dim = int(scene["meta"]["dimensions"])
    time_cfg = scene.get("time", {})
    domain_cfg = scene.get("domain", {})
    boundary_cfg = scene.get("boundary", {})
    g = np.array(scene.get("forces", {}).get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)
    rho0 = float(scene["material"]["rho0"])
    eos_k = float(scene.get("material", {}).get("eos", {}).get("k", 500.0))
    eos_c0 = scene.get("material", {}).get("eos", {}).get("c0", None)
    eos_c0 = float(eos_c0) if eos_c0 is not None else None
    visc_cfg = scene.get("material", {}).get("viscosity", {})
    neighbors_cfg = scene["neighbors"]
    support_radius = float(neighbors_cfg["support_radius"])
    smoothing_length = float(
        neighbors_cfg.get("smoothing_length") or neighbors_cfg.get("h") or support_radius / 2.0
    )
    return SimConfig(
        support_radius=support_radius,
        smoothing_length=smoothing_length,
        rho0=rho0,
        eos_k=eos_k,
        eos_c0=eos_c0,
        g=g,
        cfl_lambda=float(time_cfg.get("cfl", 0.4)),
        dt_min=float(time_cfg.get("dt_min", 1e-5)),
        dt_max=float(time_cfg.get("dt_max", 5e-4)),
        dt_fixed=float(time_cfg.get("dt_fixed", 5e-4)),
        use_cfl=(time_cfg.get("mode", "cfl") == "cfl"),
        enable_viscosity=bool(visc_cfg.get("enable", False)),
        kinematic_viscosity=float(visc_cfg.get("nu", 0.0)),
        domain_min=(np.array(domain_cfg["min"], dtype=np.float64) if "min" in domain_cfg and "max" in domain_cfg else None),
        domain_max=(np.array(domain_cfg["max"], dtype=np.float64) if "min" in domain_cfg and "max" in domain_cfg else None),
        boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),
        boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),
        boundary_eps=(
            float(boundary_cfg.get("eps"))
            if boundary_cfg.get("eps", None) is not None
            else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)
        ),
        boundary_force_accel_clamp=(
            float(boundary_cfg.get("force_accel_clamp")) if boundary_cfg.get("force_accel_clamp", None) is not None else None
        ),
    )


def run_verification(
    scene: dict,
    *,
    scene_path: str | None = None,
    report_path: str | Path | None = None,
    fast: bool = False,
) -> tuple[bool, dict]:
    """
    Deterministic verification runner with pass/fail gates.
    """
    seed = int(scene.get("meta", {}).get("seed", 0))
    np.random.seed(seed)

    if scene_path:
        scene["__scene_dir__"] = str(Path(scene_path).resolve().parent)
    state = build_scene_state(scene)
    cfg = _build_cfg(scene)
    solver_cfg = scene.get("solver", {"type": "wcsph"})

    spacing = float(scene["fluid"]["spacing"])
    h = float(scene["neighbors"]["support_radius"])
    rho0 = float(scene["material"]["rho0"])
    time_cfg = scene.get("time", {})
    verify_cfg = scene.get("verification", {})
    steps = int(time_cfg.get("steps", 100))
    every = int(verify_cfg.get("sample_every", 5))
    if fast:
        fast_cfg = verify_cfg.get("fast", {})
        steps = int(fast_cfg.get("steps", min(steps, 20)))
        every = int(fast_cfg.get("sample_every", every))

    mass0 = mass_total(state)
    p0 = momentum_total(state)
    k0 = kinetic_energy_total(state)

    records: list[dict] = []
    poiseuille_l2_hist: list[float] = []
    poiseuille_cfg = scene.get("verification", {}).get("poiseuille", {})
    do_poiseuille = bool(poiseuille_cfg.get("enable", False))
    failed_nonfinite = False
    vmax_hist: list[float] = []
    pmax_hist: list[float] = []
    dt_hist: list[float] = []

    for s in range(steps):
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict=solver_cfg,
            step_idx=s + 1,
        )
        dt_hist.append(float(dt))

        if has_non_finite_state(state):
            failed_nonfinite = True
            break

        if (s == 0) or ((s + 1) % max(1, every) == 0) or (s == steps - 1):
            fluid = state.fluid_indices
            dens = density_error_stats(state.rho[fluid], rho0=rho0)
            vmag = np.linalg.norm(state.vel[fluid], axis=1) if fluid.size else np.zeros((0,), dtype=np.float64)
            vmax = float(np.max(vmag)) if vmag.size else 0.0
            pabs_max = float(np.max(np.abs(state.p[fluid]))) if fluid.size else 0.0
            vmax_hist.append(vmax)
            pmax_hist.append(pabs_max)
            rec = {
                "step": int(s + 1),
                "dt": float(dt),
                "density_rel_err": dens,
                "vmax": vmax,
                "pabs_max": pabs_max,
                "mass": mass_total(state),
                "momentum": momentum_total(state).tolist(),
                "kinetic_energy": kinetic_energy_total(state),
            }
            if do_poiseuille:
                res = compute_vx_profile(
                    step=int(s + 1),
                    state=state,
                    scene=scene,
                    y_extent_mode=str(poiseuille_cfg.get("y_extent_mode", "walls_inner")),
                    n_bins=int(poiseuille_cfg.get("n_bins", 8)),
                    gx=float(poiseuille_cfg.get("gx", scene.get("forces", {}).get("gravity", [0.0, 0.0])[0])),
                    nu=float(poiseuille_cfg.get("nu", scene.get("material", {}).get("viscosity", {}).get("nu", 0.0))),
                )
                l2 = l2_error(res.vx_mean_per_bin, res.analytic_vx_per_bin)
                linf = linf_error(res.vx_mean_per_bin, res.analytic_vx_per_bin)
                poiseuille_l2_hist.append(l2)
                rec["poiseuille"] = {
                    "l2": l2,
                    "linf": linf,
                    "used_bins": int(res.used_bins),
                    "empty_bins": int(res.empty_bins),
                    "vmax_measured": (float(res.vmax_measured) if np.isfinite(res.vmax_measured) else None),
                    "vmax_analytic": (float(res.vmax_analytic) if res.vmax_analytic is not None else None),
                }
            records.append(rec)

    mass_final = mass_total(state)
    momentum_final = momentum_total(state)
    k_final = kinetic_energy_total(state)
    dens_mean_max = float(max((r["density_rel_err"]["mean"] for r in records), default=float("inf")))
    mass_rel_drift = float(abs(mass_final - mass0) / (abs(mass0) + 1e-12))
    mom_drift = float(np.linalg.norm(momentum_final - p0))
    kin_ratio = float(k_final / (k0 + 1e-12))

    acc = verify_cfg.get("acceptance", {})
    density_tol = float(acc.get("density_rel_err_mean_max", 0.05))
    density_max_tol = float(acc.get("density_rel_err_max_max", 1.0))
    mass_drift_tol = float(acc.get("mass_rel_drift_max", 1e-12))
    vmax_limit = acc.get("vmax_max", None)
    pabs_limit = acc.get("pabs_max", None)
    poiseuille_l2_max = float(acc.get("poiseuille_l2_max", 0.2))
    poiseuille_linf_max = float(acc.get("poiseuille_linf_max", float("inf")))
    poiseuille_empty_bins_max = int(acc.get("poiseuille_empty_bins_max", 10**9))
    poiseuille_vmax_ratio_min = acc.get("poiseuille_vmax_ratio_min", None)
    poiseuille_vmax_ratio_max = acc.get("poiseuille_vmax_ratio_max", None)
    poiseuille_vmax_ratio_min = float(poiseuille_vmax_ratio_min) if poiseuille_vmax_ratio_min is not None else None
    poiseuille_vmax_ratio_max = float(poiseuille_vmax_ratio_max) if poiseuille_vmax_ratio_max is not None else None
    require_poiseuille = bool(acc.get("require_poiseuille", False))

    poiseuille_ok = True
    steady_state = None
    poiseuille_last = None
    poiseuille_vmax_ratio = None
    if do_poiseuille and poiseuille_l2_hist:
        window = int(poiseuille_cfg.get("steady_window", 4))
        eps = float(poiseuille_cfg.get("steady_eps", 2e-3))
        if len(poiseuille_l2_hist) >= 2 * window:
            a = float(np.mean(poiseuille_l2_hist[-window:]))
            b = float(np.mean(poiseuille_l2_hist[-2 * window : -window]))
            steady_state = bool(abs(a - b) < eps)
        poiseuille_last = records[-1]["poiseuille"]
        last_l2 = float(poiseuille_last["l2"])
        last_linf = float(poiseuille_last["linf"])
        last_empty_bins = int(poiseuille_last["empty_bins"])
        vmax_measured = poiseuille_last.get("vmax_measured", None)
        vmax_analytic = poiseuille_last.get("vmax_analytic", None)
        if vmax_measured is not None and vmax_analytic is not None and np.isfinite(vmax_analytic) and abs(float(vmax_analytic)) > 1e-14:
            poiseuille_vmax_ratio = float(vmax_measured) / float(vmax_analytic)

        poiseuille_ok = bool(
            np.isfinite(last_l2)
            and np.isfinite(last_linf)
            and last_l2 <= poiseuille_l2_max
            and last_linf <= poiseuille_linf_max
            and last_empty_bins <= poiseuille_empty_bins_max
            and records[-1]["poiseuille"]["used_bins"] > 0
        )
        if poiseuille_vmax_ratio_min is not None:
            poiseuille_ok = poiseuille_ok and (poiseuille_vmax_ratio is not None and poiseuille_vmax_ratio >= poiseuille_vmax_ratio_min)
        if poiseuille_vmax_ratio_max is not None:
            poiseuille_ok = poiseuille_ok and (poiseuille_vmax_ratio is not None and poiseuille_vmax_ratio <= poiseuille_vmax_ratio_max)

    dens_max_max = float(max((r["density_rel_err"]["max"] for r in records), default=float("inf")))
    vmax_max = float(max(vmax_hist, default=0.0))
    pabs_max = float(max(pmax_hist, default=0.0))

    gates = {
        "no_non_finite": _as_gate(value=not failed_nonfinite, pass_flag=not failed_nonfinite),
        "density_rel_err_mean_max": _as_gate(
            value=dens_mean_max,
            max_value=density_tol,
            pass_flag=bool(np.isfinite(dens_mean_max) and dens_mean_max <= density_tol),
        ),
        "density_rel_err_max_max": _as_gate(
            value=dens_max_max,
            max_value=density_max_tol,
            pass_flag=bool(np.isfinite(dens_max_max) and dens_max_max <= density_max_tol),
        ),
        "mass_rel_drift_max": _as_gate(
            value=mass_rel_drift,
            max_value=mass_drift_tol,
            pass_flag=bool(np.isfinite(mass_rel_drift) and mass_rel_drift <= mass_drift_tol),
        ),
        "momentum_finite": _as_gate(value=mom_drift, pass_flag=bool(np.isfinite(mom_drift))),
        "energy_finite": _as_gate(value=kin_ratio, pass_flag=bool(np.isfinite(kin_ratio))),
        "vmax_max": _as_gate(
            value=vmax_max,
            max_value=(float(vmax_limit) if vmax_limit is not None else None),
            pass_flag=(True if vmax_limit is None else vmax_max <= float(vmax_limit)),
        ),
        "pressure_abs_max": _as_gate(
            value=pabs_max,
            max_value=(float(pabs_limit) if pabs_limit is not None else None),
            pass_flag=(True if pabs_limit is None else pabs_max <= float(pabs_limit)),
        ),
    }
    if do_poiseuille or require_poiseuille:
        last_l2 = float(poiseuille_last["l2"]) if poiseuille_last is not None else float("inf")
        last_linf = float(poiseuille_last["linf"]) if poiseuille_last is not None else float("inf")
        last_empty = int(poiseuille_last["empty_bins"]) if poiseuille_last is not None else 10**9
        gates["poiseuille_l2_max"] = _as_gate(value=last_l2, max_value=poiseuille_l2_max, pass_flag=bool(np.isfinite(last_l2) and last_l2 <= poiseuille_l2_max))
        gates["poiseuille_linf_max"] = _as_gate(value=last_linf, max_value=poiseuille_linf_max, pass_flag=bool(np.isfinite(last_linf) and last_linf <= poiseuille_linf_max))
        gates["poiseuille_empty_bins_max"] = _as_gate(value=last_empty, max_value=float(poiseuille_empty_bins_max), pass_flag=bool(last_empty <= poiseuille_empty_bins_max))
        if poiseuille_vmax_ratio_min is not None or poiseuille_vmax_ratio_max is not None:
            pass_ratio = poiseuille_vmax_ratio is not None and np.isfinite(poiseuille_vmax_ratio)
            if pass_ratio and poiseuille_vmax_ratio_min is not None:
                pass_ratio = pass_ratio and (poiseuille_vmax_ratio >= poiseuille_vmax_ratio_min)
            if pass_ratio and poiseuille_vmax_ratio_max is not None:
                pass_ratio = pass_ratio and (poiseuille_vmax_ratio <= poiseuille_vmax_ratio_max)
            gates["poiseuille_vmax_ratio"] = _as_gate(
                value=(float(poiseuille_vmax_ratio) if poiseuille_vmax_ratio is not None else float("nan")),
                min_value=poiseuille_vmax_ratio_min,
                max_value=poiseuille_vmax_ratio_max,
                pass_flag=bool(pass_ratio),
            )
        gates["poiseuille_ok"] = _as_gate(value=poiseuille_ok, pass_flag=poiseuille_ok)

    checks = {k: bool(v["pass"]) for k, v in gates.items()}
    passed = bool(all(checks.values()))

    report = {
        "scene_name": str(scene.get("meta", {}).get("name", "unknown")),
        "solver": str(solver_cfg.get("type", "wcsph")).lower(),
        "seed": seed,
        "steps": steps,
        "sample_every": every,
        "overall_pass": passed,
        "checks": checks,
        "gates": gates,
        "summary": {
            "density_rel_err_mean_max": dens_mean_max,
            "density_rel_err_max_max": dens_max_max,
            "mass_rel_drift": mass_rel_drift,
            "momentum_drift_norm": mom_drift,
            "kinetic_energy_ratio": kin_ratio,
            "vmax_max": vmax_max,
            "pabs_max": pabs_max,
            "dt_min": float(min(dt_hist, default=0.0)),
            "dt_mean": float(np.mean(dt_hist) if dt_hist else 0.0),
            "dt_max": float(max(dt_hist, default=0.0)),
            "poiseuille_l2_last": float(poiseuille_l2_hist[-1]) if poiseuille_l2_hist else None,
            "poiseuille_linf_last": (float(poiseuille_last["linf"]) if poiseuille_last is not None else None),
            "poiseuille_empty_bins_last": (int(poiseuille_last["empty_bins"]) if poiseuille_last is not None else None),
            "poiseuille_vmax_ratio_last": poiseuille_vmax_ratio,
            "poiseuille_steady": steady_state,
        },
        "records": records,
    }
    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return (passed, report)

