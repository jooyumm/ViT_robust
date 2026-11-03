"""Shared fig export and image display helpers."""
import numpy as np
import matplotlib.pyplot as plt

def save_fig(fig, path, dpi=160):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def unnormalize(image, mean=(.5,.5,.5), std=(.5,.5,.5)):
    return np.clip(image.transpose(1,2,0) * np.asarray(std) + np.asarray(mean), 0, 1)
