"""Top-level package for OlfTrans."""

__author__ = """Tingkai Liu"""
__email__ = 'tl2747@columbia.edu'
__version__ = '0.1.0'

import os
ROOTDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
KERNEL_DIR = os.path.join(ROOTDIR, 'olftrans', 'neurodriver', 'NK_kernels')
DATADIR = os.path.join(ROOTDIR, 'data')