"""Definitions for the `Blank` engine class."""
import numpy as np

from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Blank(Engine):
    """No-op engine that returns zero luminosities (e.g. for emulator SEDs)."""

    def process(self, **kwargs):
        """Process module."""
        times = kwargs[self.key('dense_times')]
        luminosities = np.zeros_like(times)

        luminosities[np.isnan(luminosities)] = 0.0

        return {self.dense_key('luminosities'): luminosities}
