# resampling_container

A lightweight Python library for jackknife and bootstrap resampling of numerical data.

The library provides three classes:

- `Jackknife`: Performs leave-one-out jackknife resampling.
- `Bootstrap`: Performs bootstrap resampling with replacement.

It supports NumPy arrays, complex-valued data, arithmetic operations, and saving/loading resampled data.

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

Let's prepare binned data.

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

### Jackknife

```python
jk = Jackknife(data)

print("Number of samples:", len(jk))
print("Central value:", jk.centre)
print("Statistical error:", jk.error)
```

For $N$ original bins, each jackknife sample excludes one bin.

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

Each bootstrap replica draws $N$ bins **with replacement** from the original $N$ bins and computes their mean. The bootstrap error is the sample standard deviation of the replicas.

The optional `rng` argument accepts a NumPy `Generator` for reproducible resampling.

# Accessing resampled data

Both `Jackknife` and `Bootstrap` allow access to individual resampled values through `.samples`, indexing, and iteration.

### Samples and indexing

```python
jk = Jackknife(data)

# All resampled samples (a list of NumPy arrays)
print(jk.samples)

# First resampled sample
print(jk.samples[0])
print(jk[0])  # Equivalent

# Number of resampled samples and shape of each sample
print(len(jk))
print(jk.shape)
```

The same operations are available for `Bootstrap` objects.

### Iteration

```python
# Iterate over all jackknife samples
for sample in jk:
    print(sample)

# Index and sample together
for i, sample in enumerate(bs):
    print(i, sample)
```

Each iteration yields a NumPy array representing one resampled sample.


## Arithmetic operations

Both `Jackknife` and `Bootstrap` support elementwise addition, subtraction, multiplication, and division, including operations with numerical scalars.
Resampling is only done at the first step (initialization of the first samples), and propagation is hadled correctly.

```python
jk1 = Jackknife(data)
jk2 = Jackknife(data)

result = 2.0 * jk1 + jk2
print("Central value:", result.centre)
print("Statistical error:", result.error)
#result = (2.0 * jk1) / jk2
#print("Central value:", result.centre)
#print("Statistical error:", result.error)
```

Arithmetic is applied to corresponding resampled values and central values.

For arithmetic between two bootstrap objects, use generators initialized with the same state, and ensure the input bins have the same count and ordering:

```python
rng1 = np.random.default_rng(1234)
rng2 = np.random.default_rng(1234)

bs1 = Bootstrap(data, nboot=5000, rng=rng1)
bs2 = Bootstrap(data, nboot=5000, rng=rng2)

result = bs1 / bs2
```

The implementation checks that bootstrap replica counts, shapes, and stored RNG states match. Matching RNG states alone does not verify that the original bins correspond.

## FILE I/O

Resampled data can be saved in NumPy `.npz` format.

### Jackknife

```python
jk.save("jackknife.npz")
jk_loaded = Jackknife.load("jackknife.npz")
```

### Bootstrap

```python
bs.save("bootstrap.npz")
bs_loaded = Bootstrap.load("bootstrap.npz")
```

Bootstrap files also store the RNG state captured before resampling.

## Complex-valued data

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

Errors for the real and imaginary components are calculated separately and returned as `error_real + 1j * error_imag`.

## Version
- 0.0.0: initial release
