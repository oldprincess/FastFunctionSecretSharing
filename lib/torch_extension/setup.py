from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import sys

cxx_flags = []
if sys.platform == "linux":
    cxx_flags = ["-maes", "-std=c++17", "-O3"]
else:
    cxx_flags = ["/std:c++17", "/O2"]

setup(
    name="pyFastFss",
    version="0.0.202504271418",
    description="Fast Function Secret Sharing (Dpf and Dcf)",
    long_description="",
    author="oldprincess",
    author_email="zirui.gong@foxmail.com",
    packages=find_packages(exclude=("test", "examples")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="pyFastFss" + "._C",
            sources=[
                "pyFastFss/src/pyFastDcf.cpp",
                "pyFastFss/src/pyFastDcfMIC.cpp",
                "pyFastFss/src/pyFastFss.cpp",
                "pyFastFss/src/pyFastGrotto.cpp",
                "pyFastFss/src/pyFastOnehot.cpp",
                "pyFastFss/src/pyFastPrng.cpp",
                "../../src/cpu/dcf.cpp",
                "../../src/cpu/grotto.cpp",
                "../../src/cpu/mic.cpp",
                "../../src/cpu/onehot.cpp",
                "../../src/cpu/prng.cpp",
                "../../src/cuda/dcf.cu",
                "../../src/cuda/grotto.cu",
                "../../src/cuda/mic.cu",
                "../../src/cuda/onehot.cu",
                "../../src/cuda/prng.cu",
            ],
            include_dirs=["../../include"],
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
