import numpy as np
import numpy.typing as npt


def clamp_step(delta: npt.NDArray[np.float64], max_magnitude: float) -> npt.NDArray[np.float64]:
    """Scales `delta` down (preserving direction) so its norm doesn't exceed max_magnitude."""
    norm = float(np.linalg.norm(delta))
    if norm <= max_magnitude or norm == 0.0:
        return delta
    return delta * (max_magnitude / norm)
