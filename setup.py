import os
from setuptools import setup
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    torch = None

def get_extensions():
    if torch is None:
        return []
    return [
        CUDAExtension('trapezoid_filter', [
            'binding.cpp',
            'trapezoid_filter.cu',
        ])
    ]

setup(
    name='trapezoid_filter',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    } if torch is not None else {}
)
