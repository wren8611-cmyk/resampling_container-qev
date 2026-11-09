import copy
import json
import numpy as np
from .container import Container

class Bootstrap(Container):
    def __init__(self, data, nboot=1000, rng=None):

        if isinstance(data, list):
            data = Container(data)

        if isinstance(data, Bootstrap):
            samples = [x.copy() for x in data.samples]
            centres = ( 
                [x.copy() for x in data.centres]
                if data.centres is not None else None
            )
            rng_state = copy.deepcopy(data.rng_state)

        elif isinstance(data, Container):
            n = len(data)

            if n < 2:
                raise ValueError("Bootstrap requires at least two bins.")

            if isinstance(nboot, (bool, np.bool_)) or not isinstance(nboot, (int, np.integer)) or nboot < 2:
                raise ValueError("nboot must be an integer >= 2.")

            if rng is None:
                rng = np.random.default_rng()

            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator.")

            # Save RNG state before resampling
            rng_state = copy.deepcopy(rng.bit_generator.state)

            centres = [np.mean(data.samples, axis=0)]

            samples = []
            for _ in range(nboot):
                indices = rng.integers(0, n, size=n)
                samples.append(
                    np.mean([data.samples[i] for i in indices], axis=0)
                )

        else:
            raise TypeError("Input must be a list, Container or Bootstrap.")

        super().__init__(samples, centres)
        self.rng_state = rng_state

    @classmethod
    def from_samples(cls, samples, centres=None, rng_state=None):
        obj = cls.__new__(cls)
        Container.__init__(obj, samples, centres)
        obj.rng_state = copy.deepcopy(rng_state)
        return obj

    @property
    def error(self):
        n = len(self)

        if n < 2:
            raise ValueError("Bootstrap requires at least two samples.")

        samples    = np.asarray(self.samples)
        error_real = np.std(samples.real, axis=0, ddof=1)

        if np.iscomplexobj(samples):
            error_imag = np.std(samples.imag, axis=0, ddof=1)
            return error_real + 1j * error_imag

        return error_real
    
    @property
    def centre(self):
        return self.centres[0] if self.centres is not None else None

    # Arithmetic operations
    def _binary_op(self, other, op):

        if isinstance(other, Bootstrap):
        
            if len(self) != len(other):
                raise ValueError("Bootstrap sample counts must match.")

            if self.shape != other.shape:
                raise ValueError("Bootstrap sample shapes must match.")

            if self.rng_state is None or other.rng_state is None:
                raise ValueError("Bootstrap RNG state is missing.")

            if self.rng_state != other.rng_state:
                raise ValueError("Bootstrap RNG states must match.")

            samples = [op(x, y) for x, y in zip(self.samples, other.samples)]

            if self.centres is not None and other.centres is not None:
            
                if len(self.centres) != len(other.centres):
                    raise ValueError("Bootstrap centre counts must match.")

                centres = [op(x, y) for x, y in zip(self.centres, other.centres)]

            else:
                centres = None

            rng_state = copy.deepcopy(self.rng_state)

        elif isinstance(other, (int, float, complex, np.number)):

            samples = [op(x, other) for x in self.samples]
            centres = (
                [op(x, other) for x in self.centres]
                if self.centres is not None else None
            )
            rng_state = copy.deepcopy(self.rng_state)

        else:
            raise TypeError(
                f"Unsupported operand type: {type(other).__name__}"
            )

        return type(self).from_samples(samples, centres, rng_state)

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
            samples=np.asarray(self.samples),
            centres=(
                np.asarray(self.centres)
                if self.centres is not None
                else np.empty((0, *self.shape))
            ),
            rng_state=json.dumps(self.rng_state),
        )

    @classmethod
    def load(cls, filename):

        with np.load(filename, allow_pickle=False) as data:
            samples = data["samples"]
            centres = data["centres"]
            rng_state = json.loads(data["rng_state"].item())

        centres = None if len(centres) == 0 else centres

        return cls.from_samples(samples, centres, rng_state)