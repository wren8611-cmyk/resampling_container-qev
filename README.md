# resampling_container

A lightweight Python library for jackknife and bootstrap resampling of numerical data.

The library provides three classes:

- `Container`: Stores numerical data as a collection of bins.
- `Jackknife`: Performs leave-one-out jackknife resampling.
- `Bootstrap`: Performs normal or balanced bootstrap resampling.

It supports NumPy arrays, multidimensional and complex-valued data, arithmetic operations, and saving/loading resampled data.

## Requirements

- Python >= 3.10
- NumPy >= 1.23

## Installation

Clone the repository and install the package:

```bash
git clone <repository-url>
cd resampling_container
python -m pip install -e .
```

## Basic usage

Prepare a collection of binned data:

```python
import numpy as np
from resampling_container import Jackknife, Bootstrap

rng = np.random.default_rng(1234)

nbins = 100
data = [
    rng.normal(loc=2.0, scale=3.0, size=5)
    for _ in range(nbins)
]
```

Each element of `data` represents one bin containing a NumPy array of shape `(5,)`.

### Jackknife

```python
jk = Jackknife(data)

print("Number of samples:", len(jk))
print("Central value:", jk.centre)
print("Statistical error:", jk.error)
```

For $N$ original bins, the jackknife procedure constructs $N$ leave-one-out samples:

$$
x_{(i)} = \frac{1}{N-1}\sum_{j\ne i}x_j.
$$

The statistical error is estimated as

$$
\sigma_{\mathrm{JK}} =
\sqrt{\frac{N-1}{N}\sum_{i=1}^{N}
\left(x_{(i)}-\overline{x}_{\mathrm{JK}}\right)^2},
$$

where $\overline{x}_{\mathrm{JK}}$ is the average of the jackknife samples.

### Bootstrap

```python
bs = Bootstrap(
    data,
    nboot=5000,
    rng=np.random.default_rng(5678),
)

print("Number of samples:", len(bs))
print("Central value:", bs.centre)
print("Statistical error:", bs.error)
```

The bootstrap procedure constructs `nboot` replicas by resampling the original bins and calculating their mean.

Two sampling methods are available:

- **`"balanced"` (default):** Constructs replicas such that every original bin appears exactly `nboot` times across all replicas.
- **`"normal"`:** Independently draws $N$ bins with replacement for each replica.

To select normal bootstrap sampling:

```python
bs = Bootstrap(
    data,
    nboot=5000,
    rng=np.random.default_rng(5678),
    method="normal",
)
```

The bootstrap statistical error is estimated using the sample standard deviation of the replicas:

$$
\sigma_{\mathrm{BS}} =
\sqrt{\frac{1}{B-1}\sum_{i=1}^{B}
\left(x_{i}^{*}-\overline{x}^{*}\right)^2},
$$

where $B$ is the number of bootstrap replicas.

The optional `rng` argument accepts a NumPy `Generator` using the PCG64 bit generator, allowing reproducible resampling.

Bootstrap objects also provide the replica average and estimated bias:

```python
print("Central value:", bs.centre)
print("Bootstrap average:", bs.average)
print("Estimated bias:", bs.bias)
```

Here, `centre` is the mean of the original bins, while `average` is the mean of the bootstrap replicas. The bias is defined as `average - centre`.

## Input data and bin structure

The library supports one-dimensional NumPy arrays and lists of numerical arrays.

For one-dimensional data, a NumPy array can be passed directly:

```python
data = np.random.normal(size=100)

jk = Jackknife(data)
bs = Bootstrap(data)
```

Each element is interpreted as one bin.

For multidimensional data, the bin direction must be specified explicitly by passing a list of arrays:

```python
data = np.random.normal(size=(100, 5, 3))

jk = Jackknife(list(data))
bs = Bootstrap(list(data))
```

In this example, there are 100 bins, each containing an array of shape `(5, 3)`.

Passing a multidimensional NumPy array directly is not supported:

```python
jk = Jackknife(data)  # Raises ValueError
bs = Bootstrap(data)  # Raises ValueError
```

This restriction avoids ambiguity in the interpretation of the bin direction.

## Accessing resampled data

Both `Jackknife` and `Bootstrap` provide access to individual resampled values through `.samples`, indexing, and iteration.

### Samples and indexing

```python
jk = Jackknife(list(data))

# All resampled samples
print(jk.samples)

# First resampled sample
print(jk.samples[0])
print(jk[0])  # Equivalent

# Number of samples and shape of each sample
print(len(jk))
print(jk.shape)
```

Each resampled sample is stored as a NumPy array. The same operations are available for `Bootstrap` objects.

### Iteration

```python
for sample in jk:
    print(sample)

for i, sample in enumerate(bs):
    print(i, sample)
```

Additional properties include:

```python
print(jk.size)
print(jk.dtype)

print(bs.nbin)
print(bs.nboot)
print(bs.method)
```

## Arithmetic operations

Both `Jackknife` and `Bootstrap` support elementwise addition, subtraction, multiplication, and division, including operations with numerical scalars.

Resampling is performed during initialization. Subsequent arithmetic operations are applied directly to corresponding resampled values, preserving the statistical fluctuations.

### Jackknife arithmetic

```python
jk1 = Jackknife(list(data))
jk2 = Jackknife(list(data))

result = 2.0 * jk1 + jk2

print("Central value:", result.centre)
print("Statistical error:", result.error)
```

Nonlinear operations are also supported:

```python
result = jk1 / jk2
```

The central value is calculated independently from the central values of the operands.

For arithmetic between two jackknife objects, the number of samples and their shapes must match. Corresponding samples must originate from the same bin ordering to preserve correlations.

### Bootstrap arithmetic

For arithmetic between two bootstrap objects, the same resampling indices must be used.

This can be achieved by initializing independent generators with identical states:

```python
rng1 = np.random.default_rng(1234)
rng2 = np.random.default_rng(1234)

bs1 = Bootstrap(list(data), nboot=5000, rng=rng1)
bs2 = Bootstrap(list(data), nboot=5000, rng=rng2)

result = bs1 / bs2

print("Central value:", result.centre)
print("Statistical error:", result.error)
```

The implementation checks that the following properties match:

- Number of original bins
- Number of bootstrap replicas
- Sample shapes
- Bootstrap sampling methods
- Stored RNG states

Matching RNG states alone does not guarantee that the original bins correspond. The input data must have the same bin ordering.

## Complex-valued data

The library supports complex-valued numerical data.

```python
rng = np.random.default_rng(1234)

data_complex = [
    rng.normal(size=5) + 1j * rng.normal(size=5)
    for _ in range(100)
]

jk_complex = Jackknife(data_complex)
bs_complex = Bootstrap(data_complex, nboot=5000)

print("Jackknife error:", jk_complex.error)
print("Bootstrap error:", bs_complex.error)
```

Statistical errors for the real and imaginary components are calculated separately and returned as

$$
\sigma = \sigma_{\mathrm{Re}} + i\sigma_{\mathrm{Im}}.
$$

The real and imaginary components of this result represent separate statistical uncertainties, rather than a complex-valued standard deviation.

## File I/O

Resampled data can be saved and loaded using NumPy's `.npz` format.

### Jackknife

```python
jk.save("jackknife.npz")

jk_loaded = Jackknife.load("jackknife.npz")

print(jk_loaded.centre)
print(jk_loaded.error)
```

### Bootstrap

```python
bs.save("bootstrap.npz")

bs_loaded = Bootstrap.load("bootstrap.npz")

print(bs_loaded.centre)
print(bs_loaded.error)
```

Bootstrap files additionally store the number of original bins, the sampling method, and the RNG state captured before resampling.

The original binned data are not stored in the resampling files.

## Tutorial

A more comprehensive example is available in:

```text
tutorial/sample_code.py
```

Run it using:

```bash
python tutorial/sample_code.py
```

The tutorial demonstrates resampling, arithmetic operations, complex-valued data, multidimensional samples, reproducibility, and file I/O.

## Version

- **0.0.0** — Initial release.
