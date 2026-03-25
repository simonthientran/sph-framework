#!/usr/bin/env python3
"""
Profile the new vectorized SPH implementation.
"""
import cProfile
import pstats
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sph.simulator_new import Simulator

def run_simulation():
    scene_path = Path(__file__).parent.parent / "scenes" / "examples" / "pipe_flow_2d.json"
    sim = Simulator(scene_path)

    # Run 5 steps
    for _ in range(5):
        sim.step()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_simulation()

    profiler.disable()

    # Print top 20 functions by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    stats.print_stats(20)
