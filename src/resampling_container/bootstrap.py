import copy as cp
import json
import numpy as np
from .container import Container

class Bootstrap(Container):
    def __init__(self, data, nboot=256, rng=None, method="balanced"):
        self.method = method

        if isinstance(data, (np.ndarray, list)):
            data = Container(data)

        if type(data) is Container:
            self.nbin_  = len(data)

            if self.nbin_ < 2:
                raise ValueError("Bootstrap requires at least two bins.")

            if isinstance(nboot, (bool, np.bool_)) or not isinstance(nboot, (int, np.integer)) or nboot < 2:
                raise ValueError("nboot must be an integer >= 2.")

            if rng is None:
                rng = np.random.default_rng()
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator.")
            if not isinstance(rng.bit_generator, np.random.PCG64):
                raise TypeError("Only PCG64 bit generators are supported.")
            rng_state = cp.deepcopy(rng.bit_generator.state)

            centre = data.eval_mean()
            if method == "normal":
                samples = self._bootstrap_sampling_normal(data, nboot, rng)
            elif method == "balanced":
                samples = self._bootstrap_sampling_balanced(data, nboot, rng)
            else:
                raise ValueError(f"Unsupported bootstrap method: {method}")

        else:
            raise TypeError("Input must be a list, np.ndarray, Container.")

        super().__init__(samples, centre, copy=False)
        self.rng_state = rng_state

    @classmethod
    def from_samples(cls, samples, centre=None, *, rng_state, nbin, method, copy=True):
        obj = cls.__new__(cls)
        Container.__init__(obj, samples, centre, copy=copy)

        obj.nbin_     = nbin
        obj.rng_state = cp.deepcopy(rng_state)
        obj.method    = method
        return obj
    
    @property
    def average(self):
        return self.eval_mean()
    
    @property
    def bias(self):
        if self.centre is None:
            return None
        return self.average - self.centre

    @property
    def error(self):
        if len(self) < 2:
            raise ValueError("Bootstrap requires at least two samples.")

        samples    = np.asarray(self.samples)
        error_real = np.std(samples.real, axis=0, ddof=1)

        if np.iscomplexobj(samples):
            error_imag = np.std(samples.imag, axis=0, ddof=1)
            return error_real + 1j * error_imag

        return error_real

    # Arithmetic operations
    def _binary_op(self, other, op):
        if isinstance(other, Bootstrap):
            if self.nbin_ != other.nbin_:
                raise ValueError("Bootstrap original bin counts must match.")
            
            if self.method != other.method:
                raise ValueError("Bootstrap sampling methods must match.")
            
            if len(self) != len(other):
                raise ValueError("Bootstrap sample counts must match.")

            if self.shape != other.shape:
                raise ValueError("Bootstrap sample shapes must match.")

            if self.rng_state is None or other.rng_state is None:
                raise ValueError("Bootstrap RNG state is missing.")

            if self.rng_state != other.rng_state:
                raise ValueError("Bootstrap RNG states must match.")

            samples   = [op(x, y) for x, y in zip(self.samples, other.samples)]
            centre    = op(self.centre, other.centre) if self.centre is not None and other.centre is not None else None

        elif isinstance(other, (int, float, complex, np.number)):
            samples   = [op(x, other) for x in self.samples]
            centre    = op(self.centre, other) if self.centre is not None else None

        else:
            raise TypeError(f"Unsupported operand type: {type(other).__name__}")

        return type(self).from_samples(samples, centre, rng_state=self.rng_state, nbin=self.nbin_, method=self.method, copy=False)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return self._binary_op(other, lambda x, y: np.subtract(y, x))

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda x, y: np.divide(y, x))

    def __neg__(self):
        return self * (-1)
    
    # io
    def save(self, filename):
        np.savez(
            filename,
            samples    = np.asarray(self.samples),
            centre     = self.centre if self.centre is not None else np.empty(0),
            has_centre = self.centre is not None,
            rng_state  = json.dumps(self.rng_state),
            nbin       = self.nbin_,
            method     = self.method,
        )

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as data:
            samples   = data["samples"]
            centre    = data["centre"] if bool(data["has_centre"]) else None
            rng_state = json.loads(data["rng_state"].item())
            nbin      = int(data["nbin"])
            method    = str(data["method"].item())
    
        return cls.from_samples(list(samples), centre, rng_state=rng_state, nbin=nbin, method=method)
    
    @property
    def nbin(self):
        return self.nbin_

    @property
    def nboot(self):
        return len(self)
    
    @staticmethod
    def _bootstrap_sampling_normal(data, nboot, rng):
        nbin    = len(data)
        array   = np.asarray(data.samples) 
        samples = []
        for _ in range(nboot):
            indices = rng.integers(0, nbin, size=nbin)
            samples.append(np.mean(array[indices], axis=0))

        return samples

    @staticmethod
    def _bootstrap_sampling_balanced(data, nboot, rng):
        nbin    = len(data)
        array   = np.asarray(data.samples)
        samples = []

        indices = np.tile(np.arange(nbin), nboot)
        rng.shuffle(indices)
        indices = indices.reshape(nboot, nbin)

        for idx in indices:
            samples.append(np.mean(array[idx], axis=0))

        return samples