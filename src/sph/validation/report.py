"""Validation utilities for interactive diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sph.core.diagnostics import StepDiagnostics
from sph.validation.profile import VelocityProfile


Classification = Literal["pass", "warn", "fail"]


@dataclass
class ValidationReport:
    diagnostics: StepDiagnostics
    profile: VelocityProfile
    rho0: float
    density_error_mean: float
    stability: Classification

    @classmethod
    def from_diagnostics(
        cls,
        diagnostics: StepDiagnostics,
        profile: VelocityProfile,
        rho0: float,
        rho_tol: float = 0.02,
        neighbor_tol: int = 10,
    ) -> "ValidationReport":
        density_error_mean = float(abs(diagnostics.rho_rel_err_mean))
        stability: Classification
        if diagnostics.n_fluid == 0:
            stability = "fail"
        elif diagnostics.neigh_min < max(1, neighbor_tol // 2):
            stability = "fail"
        elif density_error_mean > rho_tol * 1.5:
            stability = "fail"
        elif density_error_mean > rho_tol or diagnostics.neigh_min < neighbor_tol:
            stability = "warn"
        else:
            stability = "pass"
        return cls(
            diagnostics=diagnostics,
            profile=profile,
            rho0=float(rho0),
            density_error_mean=density_error_mean,
            stability=stability,
        )

    @property
    def summary(self) -> str:
        diag = self.diagnostics
        return (
            f"[{self.stability.upper()}] step={diag.step} dt={diag.dt:.3e} "
            f"rho(avg)={diag.rho_mean:.2f} err%={self.density_error_mean * 100.0:.2f} "
            f"neighbors={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max} "
            f"|v|max={diag.v_max:.3f}"
        )

    def to_dict(self) -> dict:
        """Serialize diagnostics for CLI benchmarks or logging."""

        diag = self.diagnostics
        profile_dict = {
            "y_centers": self.profile.y_centers.tolist(),
            "vx_avg": self.profile.vx_avg.tolist(),
            "pipe_height": float(self.profile.pipe_height),
            "centerline_vx": float(self.profile.centerline_velocity),
            "vmax": float(self.profile.vmax),
        }
        return {
            "step": int(diag.step),
            "dt": float(diag.dt),
            "rho_min": float(diag.rho_min),
            "rho_mean": float(diag.rho_mean),
            "rho_max": float(diag.rho_max),
            "rho_rel_err_mean": float(diag.rho_rel_err_mean),
            "p_min": float(diag.p_min),
            "p_mean": float(diag.p_mean),
            "p_max": float(diag.p_max),
            "neighbor_min": int(diag.neigh_min),
            "neighbor_mean": float(diag.neigh_mean),
            "neighbor_max": int(diag.neigh_max),
            "velocity_max": float(diag.v_max),
            "density_error_mean": float(self.density_error_mean),
            "stability": self.stability,
            "velocity_profile": profile_dict,
        }
