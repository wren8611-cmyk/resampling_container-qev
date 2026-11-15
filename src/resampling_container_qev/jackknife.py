import numpy as np
from .container import Container

class Jackknife(Container):
    def __init__(self, data):
        if isinstance(data, (np.ndarray, list)):
            data = Container(data)

        if type(data) is Container:
            n = len(data)
            if n < 2:
                raise ValueError("Jackknife requires at least two bins.")

            total   = np.sum(data.samples, axis=0)
            samples = [(total - x) / (n - 1) for x in data.samples]
            centre  = total / n

        else:
            raise TypeError("Input must be a list, np.ndarray, or Container.")

        super().__init__(samples, centre, copy=False)

    @classmethod
    def from_samples(cls, samples, centre=None, *, copy=True):
        obj = cls.__new__(cls)
        Container.__init__(obj, samples, centre, copy=copy)
        return obj

    @property
    def error(self):
        n = len(self)
        if n < 2:
            raise ValueError("Jackknife requires at least two samples.")

        samples = np.asarray(self.samples)
        mean    = np.mean(samples, axis=0)
        diff    = samples - mean

        error_real = np.sqrt((n - 1) / n * np.sum(diff.real**2, axis=0))

        if np.iscomplexobj(samples):
            error_imag = np.sqrt((n - 1) / n * np.sum(diff.imag**2, axis=0))
            return error_real + 1j * error_imag

        return error_real

    # Arithmetic operations
    def _binary_op(self, other, op):
        if isinstance(other, Jackknife):
            if len(self) != len(other):
                raise ValueError("Jackknife sample counts must match.")
            if self.shape != other.shape:
                raise ValueError("Jackknife sample shapes must match.")

            samples = [op(x, y) for x, y in zip(self.samples, other.samples)]
            centre  = op(self.centre, other.centre) if self.centre is not None and other.centre is not None else None

        elif isinstance(other, (int, float, complex, np.number)):
            samples = [op(x, other) for x in self.samples]
            centre  = op(self.centre, other) if self.centre is not None else None

        else:
            raise TypeError(f"Unsupported operand type: {type(other).__name__}")

        return type(self).from_samples(samples, centre, copy=False)

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
        )

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as data:
            samples = data["samples"]
            centre  = data["centre"] if bool(data["has_centre"]) else None

        return cls.from_samples(list(samples), centre)