import os

# Enable Numba's CUDA simulator before any test module imports numba.cuda.
# Without a physical GPU, numba.cuda otherwise raises CUDA_ERROR_NO_DEVICE.
# Setting this here (conftest is imported before test collection) guarantees
# it takes effect ahead of the first numba.cuda import in any test module.
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

collect_ignore_glob = ["sph-framework_backup/**"]
