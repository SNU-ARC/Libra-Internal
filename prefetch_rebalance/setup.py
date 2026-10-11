'''
Usage: python setup.py build_ext --inplace
'''

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="prefetch_rebalance_v8_cython",
    sources=["prefetch_rebalance_v8_cython.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[
        ('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION'),
    ],
    extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
)

setup(
    ext_modules=cythonize([ext]),
)
