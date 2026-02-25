from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

class FlowMetrics:
    def __init__(self, output_dir: str | Path, scene_name: str, config: dict):
        self.output_dir = Path(output_dir) / scene_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.csv"
        
        self.planes = config.get("planes", [])
        self.profiles = config.get("profiles", [])
        
        self.headers = [
            "step", "time", "dt",
            "rho_min", "rho_mean", "rho_max",
            "v_max", "active_count", "inactive_count"
        ]
        for p in self.planes:
            self.headers.append(f"mass_flow_{p['name']}")
        for pf in self.profiles:
            for b in range(pf.get("bins", 10)):
                self.headers.append(f"vel_prof_{pf['name']}_bin{b}")
                
        # Write header
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, time: float, dt: float, state):
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        pos = state.pos[fluid_ids]
        vel = state.vel[fluid_ids]
        rho = state.rho[fluid_ids]
        mass = state.mass[fluid_ids]
        
        active_mask = pos[:, 0] < 1e8
        active_ids = fluid_ids[active_mask]
        inactive_ids = fluid_ids[~active_mask]
        
        active_pos = state.pos[active_ids]
        active_vel = state.vel[active_ids]
        active_rho = state.rho[active_ids]
        active_mass = state.mass[active_ids]

        if active_pos.size == 0:
            return

        rho_min = float(np.min(active_rho))
        rho_mean = float(np.mean(active_rho))
        rho_max = float(np.max(active_rho))
        v_max = float(np.max(np.linalg.norm(active_vel, axis=1)))
        
        row = [
            step, time, dt,
            rho_min, rho_mean, rho_max, v_max,
            len(active_ids), len(inactive_ids)
        ]
        
        # Mass flow planes
        for p_cfg in self.planes:
            pmin = np.array(p_cfg["min"])
            pmax = np.array(p_cfg["max"])
            normal = np.array(p_cfg["normal"])
            
            in_plane = (active_pos >= pmin) & (active_pos <= pmax)
            in_plane_mask = np.all(in_plane, axis=1)
            
            if np.any(in_plane_mask):
                plane_vel = active_vel[in_plane_mask]
                plane_mass = active_mass[in_plane_mask]
                v_n = plane_vel @ normal
                mass_flow = np.sum(plane_mass * v_n)
                row.append(float(mass_flow))
            else:
                row.append(0.0)
                
        # Velocity profiles
        for pf_cfg in self.profiles:
            pmin = np.array(pf_cfg["min"])
            pmax = np.array(pf_cfg["max"])
            bins = int(pf_cfg.get("bins", 10))
            axis = int(pf_cfg.get("axis", 1)) # Default slice along y
            
            in_region = (active_pos >= pmin) & (active_pos <= pmax)
            in_region_mask = np.all(in_region, axis=1)
            
            if np.any(in_region_mask):
                region_pos = active_pos[in_region_mask]
                region_vel = active_vel[in_region_mask]
                
                y0 = float(pmin[axis])
                y1 = float(pmax[axis])
                edges = np.linspace(y0, y1, bins + 1)
                
                for b in range(bins):
                    m = (region_pos[:, axis] >= edges[b]) & (region_pos[:, axis] < edges[b+1])
                    if np.any(m):
                        # typically we want streamwise velocity, assume x (0)
                        mean_vx = np.mean(region_vel[m, 0])
                        row.append(float(mean_vx))
                    else:
                        row.append(0.0)
            else:
                for b in range(bins):
                    row.append(0.0)
                    
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
