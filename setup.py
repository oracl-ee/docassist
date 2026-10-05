from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# /O2 = optimize for speed, /openmp = multithreading
ext_modules = [
    Pybind11Extension("fastsearch", ["cpp/fastsearch.cpp"],
                      cxx_std=17, extra_compile_args=["/O2", "/openmp"]),
]

setup(name="fastsearch", ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext})