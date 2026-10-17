"""Convenience root wrapper for fillers/thesis_filler_experiments.py"""
import sys
from pathlib import Path

# Add fillers directory to sys.path
fillers_dir = Path(__file__).resolve().parent / "fillers"
sys.path.insert(0, str(fillers_dir))

from thesis_filler_experiments import main

if __name__ == "__main__":
    main()
