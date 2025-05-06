from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module with the correct compiler options
extension = Extension(
    name="openmp_example",  # Name of the extension module
    sources=["openmp_example.pyx"],  # Cython source file
    extra_compile_args=["-fopenmp"],  # OpenMP flags for GCC
    include_dirs=[numpy.get_include()],  # Include numpy headers
)

# Build the extension module with language_level 3
setup(
    ext_modules=cythonize([extension], compiler_directives={'language_level': "3"}),
)
