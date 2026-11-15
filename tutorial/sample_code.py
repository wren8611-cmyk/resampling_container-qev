"""
End-to-end tutorial for resampling_container.
Run after installing the package: python tutorial/sample_code.py
"""
from pathlib import Path
import numpy as np
from resampling_container import Container, Jackknife, Bootstrap


def show(label, obj):
    print(f"\n{label}: {obj!r}")
    print("  centre:", obj.centre)
    if isinstance(obj, (Jackknife, Bootstrap)):
        print("  error: ", obj.error)
    if isinstance(obj, Bootstrap):
        print("  average:", obj.average)
        print("  bias:   ", obj.bias)
        print("  method: ", obj.method)
        print("  nbin/nboot:", obj.nbin, obj.nboot)


def main():
    rng = np.random.default_rng(1234)
    nbin, nboot, seed = 24, 512, 777

    # 1. Container: a one-dimensional ndarray means scalar observations.
    scalar_data = rng.normal(loc=2.0, scale=1.0, size=nbin)
    scalar = Container(scalar_data)
    print("Container:", scalar)
    print("  size / shape / dtype:", scalar.size, scalar.shape, scalar.dtype)
    print("  first sample:", scalar[0])
    print("  iterated samples:", sum(1 for _ in scalar))
    print("  mean:", scalar.eval_mean())

    # 2. For vector/matrix observations, explicitly pass a list of bins.
    matrix_data = rng.normal(size=(nbin, 2, 3))
    matrix_bins = list(matrix_data)  # each bin is a (2, 3) matrix
    matrix = Container(matrix_bins)
    print("\nMatrix-valued Container:", matrix)
    assert matrix.shape == (2, 3)
    try:
        Container(matrix_data)
    except ValueError as exc:
        print("  Expected multidimensional ndarray rejection:", exc)
    else:
        raise AssertionError("Multidimensional ndarray should be rejected")

    # 3. Jackknife: leave-one-out means, central estimate, and standard error.
    jk = Jackknife(scalar_data)
    show("Jackknife", jk)
    np.testing.assert_allclose(jk.centre, np.mean(scalar_data))
    np.testing.assert_allclose(jk.error, np.std(scalar_data, ddof=1) / np.sqrt(nbin))
    show("Matrix jackknife", Jackknife(matrix_bins))

    # 4. Bootstrap: balanced (default) and normal sampling.
    bs = Bootstrap(scalar_data, nboot=nboot, rng=np.random.default_rng(seed))
    bs_normal = Bootstrap(scalar_data, nboot=nboot, rng=np.random.default_rng(seed), method="normal")
    show("Balanced bootstrap", bs)
    show("Normal bootstrap", bs_normal)
    np.testing.assert_allclose(bs.average, bs.centre, atol=1e-12)
    show("Matrix bootstrap", Bootstrap(matrix_bins, nboot=128, rng=np.random.default_rng(seed)))

    # 5. Scalar arithmetic and nonlinear transformations of matched replicas.
    for label, result in [
        ("jk + 2", jk + 2), ("2 + jk", 2 + jk),
        ("jk - 2", jk - 2), ("2 - jk", 2 - jk),
        ("jk * 2", jk * 2), ("2 * jk", 2 * jk),
        ("jk / 2", jk / 2), ("2 / jk", 2 / jk),
        ("-jk", -jk), ("jk * jk", jk * jk),
        ("bs + 2", bs + 2), ("2 + bs", 2 + bs),
        ("bs - 2", bs - 2), ("2 - bs", 2 - bs),
        ("bs * 2", bs * 2), ("2 * bs", 2 * bs),
        ("bs / 2", bs / 2), ("2 / bs", 2 / bs),
        ("-bs", -bs), ("bs * bs", bs * bs),
    ]:
        show(label, result)

    # Matched bootstrap objects require identical RNG starting states,
    # method, sample count, and original bin count.
    second_data = scalar_data + 1.0
    bs2 = Bootstrap(second_data, nboot=nboot, rng=np.random.default_rng(seed))
    show("Paired bootstrap ratio", bs2 / bs)
    jk2 = Jackknife(second_data)
    show("Paired jackknife ratio", jk2 / jk)
    try:
        bs + bs_normal
    except ValueError as exc:
        print("\nExpected method mismatch:", exc)
    else:
        raise AssertionError("Different bootstrap methods should not combine")

    # 6. Complex-valued data: errors for real and imaginary parts separately.
    complex_data = scalar_data + 1j * rng.normal(size=nbin)
    show("Complex jackknife", Jackknife(complex_data))
    show("Complex bootstrap", Bootstrap(complex_data, nboot=nboot, rng=np.random.default_rng(seed)))

    # 7. from_samples: reconstruct existing replicas and centre.
    jk_copy = Jackknife.from_samples(list(jk.samples), jk.centre)
    bs_copy = Bootstrap.from_samples(
        list(bs.samples), bs.centre,
        rng_state=bs.rng_state, nbin=bs.nbin, method=bs.method,
    )
    np.testing.assert_allclose(jk_copy.error, jk.error)
    np.testing.assert_allclose(bs_copy.error, bs.error)
    print("\nfrom_samples reconstruction: passed")

    # 8. NPZ round-trip, including multidimensional samples and RNG metadata.
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    jk_matrix = Jackknife(matrix_bins)
    bs_matrix = Bootstrap(matrix_bins, nboot=128, rng=np.random.default_rng(seed))
    jk_matrix.save(output_dir / "jackknife.npz")
    bs_matrix.save(output_dir / "bootstrap.npz")
    jk_loaded = Jackknife.load(output_dir / "jackknife.npz")
    bs_loaded = Bootstrap.load(output_dir / "bootstrap.npz")
    for original, loaded in [(jk_matrix, jk_loaded), (bs_matrix, bs_loaded)]:
        np.testing.assert_array_equal(np.asarray(original.samples), np.asarray(loaded.samples))
        np.testing.assert_array_equal(original.centre, loaded.centre)
        np.testing.assert_allclose(original.error, loaded.error)
    assert bs_loaded.nbin == bs_matrix.nbin
    assert bs_loaded.nboot == bs_matrix.nboot
    assert bs_loaded.method == bs_matrix.method
    assert bs_loaded.rng_state == bs_matrix.rng_state
    print("NPZ save/load (including multidimensional samples): passed")
    print("\nAll tutorial checks passed.")


if __name__ == "__main__":
    main()
