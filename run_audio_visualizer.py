#!/usr/bin/env python3
"""
Launch script for the GPU-Accelerated Audio Visualization Application.
This script should be run from the PythonProject directory.
"""

import sys
import os
from pathlib import Path

# Add the audio_visualizer module to Python path
project_root = Path(__file__).parent
audio_visualizer_path = project_root / "audio_visualizer"
sys.path.insert(0, str(project_root))

# Import and run the main application
if __name__ == "__main__":
    from audio_visualizer.main import main
    sys.exit(main())