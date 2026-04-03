from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import sys
import glob
import os
import torch


cxx_flags = []
if sys.platform == "linux":
    cxx_flags = ["-maes", "-std=c++20", "-O3", "-fopenmp", "-fvisibility=hidden"]
else:
    cxx_flags = ["/std:c++20", "/O2", "/openmp"]

if not torch.cuda.is_available():
    if sys.platform == "linux":
        cxx_flags += ["-DNO_CUDA"]
    else:
        cxx_flags += ["/DNO_CUDA"]
    ext_module = cpp_extension.CppExtension(
        name="pyFastFss" + "._C",
        sources=[
            # python
            *glob.glob("pyFastFss/src/*.cpp"),
            # public api
            *glob.glob("../../src/*.cpp"),
            # cpu
            *glob.glob("../../src/cpu/*.cpp"),
        ],
        include_dirs=[
            os.path.abspath("../../include"),
            os.path.abspath("../../third_party/wideint/include"),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
        },
    )
else:
    ext_module = cpp_extension.CUDAExtension(
        name="pyFastFss" + "._C",
        sources=[
            # python
            *glob.glob("pyFastFss/src/*.cpp"),
            # public api
            *glob.glob("../../src/*.cpp"),
            # cpu
            *glob.glob("../../src/cpu/*.cpp"),
            # cuda
            *glob.glob("../../src/cuda/*.cpp"),
            *glob.glob("../../src/cuda/*.cu"),
        ],
        include_dirs=[
            os.path.abspath("../../include"),
            os.path.abspath("../../third_party/wideint/include"),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ["-O3", "-std=c++20"],
        },
    )

setup(
    name="pyFastFss",
    version="0.0.202512061650",
    description="Fast Function Secret Sharing (Dpf and Dcf)",
    long_description="",
    author="oldprincess",
    author_email="zirui.gong@foxmail.com",
    packages=find_packages(exclude=("test", "examples")),
    ext_modules=[ext_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
