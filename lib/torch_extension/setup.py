from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import sys
import glob
import os

cxx_flags = []
if sys.platform == "linux":
    cxx_flags = ["-maes", "-std=c++17", "-O3", "-fopenmp", "-fvisibility=hidden"]
else:
    cxx_flags = ["/std:c++17", "/O2", "/openmp"]

setup(
    name="pyFastFss",
    version="0.0.202511081500",
    description="Fast Function Secret Sharing (Dpf and Dcf)",
    long_description="",
    author="oldprincess",
    author_email="zirui.gong@foxmail.com",
    packages=find_packages(exclude=("test", "examples")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="pyFastFss" + "._C",
            sources=[
                # python
                *glob.glob("pyFastFss/src/*.cpp"),
                # cpu
                *glob.glob("../../src/cpu/*.cpp"),
                # cuda
                *glob.glob("../../src/cuda/*.cpp"),
                *glob.glob("../../src/cuda/*.cu"),
            ],
            include_dirs=[os.path.abspath("../../include")],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
