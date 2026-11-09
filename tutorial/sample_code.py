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