#!/usr/bin/env python3
"""
Audiobook Generator GUI Launcher
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable TensorFloat32 for better Blackwell performance
import torch
torch.set_float32_matmul_precision('high')

from pocket_tts.gui.main_window import main

if __name__ == "__main__":
    # Deliberate multiprocessing start method: spawn is required for CUDA
    # workers on both Windows and Linux (fork inherits GPU state and crashes).
    import multiprocessing as _mp
    _mp.set_start_method('spawn', force=True)
    main()
