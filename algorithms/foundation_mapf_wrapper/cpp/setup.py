from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "mapf_features_cpp",
        ["mapf_features.cpp"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++17"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="mapf_features_cpp",
    version="1.0.0",
    author="Foundation MAPF",
    description="C++ acceleration for Foundation MAPF feature construction",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
