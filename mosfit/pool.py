"""Compatibility helpers for processing pools used by MOSFiT."""

from schwimmbad import SerialPool


class MOSFiTSerialPool(SerialPool):
    """SerialPool with the MPI-like attributes MOSFiT expects."""

    size = 0
    comm = None

    def is_master(self):
        """Return True for the sole serial process."""
        return True

    def wait(self):
        """Match MPIPool.wait() with a no-op in serial execution."""
        return None

    def close(self):
        """Match MPIPool.close() with a no-op in serial execution."""
        return None
