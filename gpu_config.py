"""
GPU setup and device memory configuration for TensorFlow.
Ensures bundled CUDA/cuDNN libraries in the virtual environment are loaded
and enables memory growth to prevent Out-Of-Memory issues on laptop GPUs.
"""

import os
import sys
import glob
import ctypes


def setup_gpu():
    """Configure CUDA dynamic libraries and setup TensorFlow GPU memory growth.

    Returns:
        list: Detected physical GPU devices.
    """
    # Preload bundled nvidia libraries in site-packages if not already loaded
    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_path):
            for lib in sorted(glob.glob(os.path.join(nvidia_path, "*", "lib", "*.so*"))):
                try:
                    ctypes.CDLL(lib)
                except Exception:
                    pass

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    return gpus
