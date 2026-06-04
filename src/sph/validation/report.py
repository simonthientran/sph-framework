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
        density_error_mean = float(abs(diagnostics.rho_error_mean))
        stability: Classification
        n_fluid = getattr(diagnostics, "n_fluid", getattr(diagnostics, "fluid_count", 0))
        neigh_min = getattr(diagnostics, "neigh_min", getattr(diagnostics, "neighbor_min", 0))
        if n_fluid == 0:
            stability = "fail"
        elif neigh_min < max(1, neighbor_tol // 2):
            stability = "fail"
        elif density_error_mean > rho_tol * 1.5:
            stability = "fail"
        elif density_error_mean > rho_tol or neigh_min < neighbor_tol:
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
            f"neighbors={getattr(diag,'neighbor_min',getattr(diag,'neigh_min',0))}/"
            f"{getattr(diag,'neighbor_mean',getattr(diag,'neigh_mean',0.0)):.1f}/"
            f"{getattr(diag,'neighbor_max',getattr(diag,'neigh_max',0))} "
            f"|v|max={getattr(diag,'velocity_max',getattr(diag,'v_max',0.0)):.3f}"
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
            "rho_error_mean": float(diag.rho_error_mean),
            "p_min": float(getattr(diag, "pressure_min", getattr(diag, "p_min", 0.0))),
            "p_mean": float(getattr(diag, "pressure_mean", getattr(diag, "p_mean", 0.0))),
            "p_max": float(getattr(diag, "pressure_max", getattr(diag, "p_max", 0.0))),
            "neighbor_min": int(getattr(diag, "neighbor_min", getattr(diag, "neigh_min", 0))),
            "neighbor_mean": float(getattr(diag, "neighbor_mean", getattr(diag, "neigh_mean", 0.0))),
            "neighbor_max": int(getattr(diag, "neighbor_max", getattr(diag, "neigh_max", 0))),
            "velocity_max": float(getattr(diag, "velocity_max", getattr(diag, "v_max", 0.0))),
            "density_error_mean": float(self.density_error_mean),
            "stability": self.stability,
            "velocity_profile": profile_dict,
        }
