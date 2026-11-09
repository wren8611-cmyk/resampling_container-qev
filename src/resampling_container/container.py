import numpy as np

class Container:
    def __init__(self, samples, centres=None):
        # Samples (mandatory)
        self.samples = [np.asarray(x) for x in samples]

        if not self.samples:
            raise ValueError("Container cannot be empty.")

        shape = self.samples[0].shape

        if any(x.shape != shape for x in self.samples):
            raise ValueError("All samples must have the same shape.")

        # Centres (optional)
        self.centres = None

        if centres is not None:
            self.centres = [np.asarray(x) for x in centres]

            if any(x.shape != shape for x in self.centres):
                raise ValueError("All centres must have the same shape as samples.")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @property
    def shape(self):
        return self.samples[0].shape

    @property
    def size(self):
        return len(self.samples)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(size={self.size}, shape={self.shape}, "
            f"ncentres={len(self.centres) if self.centres is not None else 0})"
        )
    
    def get_centres(self):
        return self.centres

    def eval_mean(self):
        if self.centres is None:
            mean = np.mean(self.samples, axis=0)
            self.centres = [mean]

        return self.centres