#!/usr/bin/env python3
"""
Simple launcher script for the Intelligent Research Assistant UI.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import main

if __name__ == "__main__":
    sys.exit(main())