# sample_code.py

import numpy as np
from resampling_container import Jackknife, Bootstrap

# Generate sample data
rng = np.random.default_rng(1234)

nbins = 100
data = [
    rng.normal(loc=2.0, scale=3.0, size=5)
    for _ in range(nbins)
]

# Jackknife
jk = Jackknife(data)

# Bootstrap
bs = Bootstrap(
    data,
    nboot=5000,
    rng=np.random.default_rng(5678)
)

# Results
print("Jackknife:")
print("  Centre:", jk.centre)
print("  Error :", jk.error )

print("\nBootstrap:")
print("  Centre:", bs.centre)
print("  Error :", bs.error )

# Arithmatic
print("\nArithmetic operations:")

# Jackknife
jk2 = 2.0 * jk + 1.0
jk3 = jk * jk

print("\nJackknife (2 * jk + 1):")
print("  Centre:", jk2.centre)
print("  Error :", jk2.error )

print("\nJackknife (jk * jk):")
print("  Centre:", jk3.centre)
print("  Error :", jk3.error )

# Bootstrap
bs2 = 2.0 * bs + 1.0
bs3 = bs * bs

print("\nBootstrap (2 * bs + 1):")
print("  Centre:", bs2.centre)
print("  Error :", bs2.error )

print("\nBootstrap (bs * bs):")
print("  Centre:", bs3.centre)
print("  Error :", bs3.error )

# File I/O
print("\nFile I/O:")

from pathlib import Path

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

# Save
jk.save(output_dir / "jackknife.npz")
bs.save(output_dir / "bootstrap.npz")

# Load
jk_loaded = Jackknife.load(output_dir / "jackknife.npz")
bs_loaded = Bootstrap.load(output_dir / "bootstrap.npz")

# Check loaded data
print("\nJackknife (loaded):")
print("  Centre:", jk_loaded.centre)
print("  Error :", jk_loaded.error)

print("\nBootstrap (loaded):")
print("  Centre:", bs_loaded.centre)
print("  Error :", bs_loaded.error)

# Verify consistency
np.testing.assert_allclose(jk_loaded.samples, jk.samples)
np.testing.assert_allclose(jk_loaded.centre, jk.centre)
np.testing.assert_allclose(jk_loaded.error, jk.error)

np.testing.assert_allclose(bs_loaded.samples, bs.samples)
np.testing.assert_allclose(bs_loaded.centre, bs.centre)
np.testing.assert_allclose(bs_loaded.error, bs.error)

assert bs_loaded.rng_state == bs.rng_state

print("\nFile I/O tests passed!")