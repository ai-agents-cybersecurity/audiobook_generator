#!/usr/bin/env python3
"""
Runner script for audiobook-generator CLI.
Use this if you haven't installed the package with pip.

Usage:
    python run.py show-config
    python run.py generate mybook.rtf
    python run.py generate-chapters mybook/
    python run.py info mybook.rtf
    python run.py voices
"""

import sys
import os

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobook_generator.cli import app

if __name__ == "__main__":
    app()
