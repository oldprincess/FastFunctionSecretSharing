from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import sys
import glob
import os
import torch

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))


def here(*parts):
    return os.path.join(THIS_DIR, *parts)


def repo(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def here_glob(*parts):
    return glob.glob(here(*parts), recursive=True)


def repo_glob(*parts):
    return glob.glob(repo(*parts), recursive=True)


cxx_flags = []
if sys.platform == "linux":
    cxx_flags = ["-maes", "-std=c++17", "-O3", "-fopenmp", "-fvisibility=hidden"]
else:
    cxx_flags = ["/std:c++17", "/O2", "/openmp"]

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-arch=native",
    "--expt-relaxed-constexpr",
    "--default-stream=per-thread",
]
if sys.platform == "win32":
    nvcc_flags += ['-Xcompiler="/EHsc"']

if not torch.cuda.is_available():
    if sys.platform == "linux":
        cxx_flags += []
    else:
        cxx_flags += []
    ext_module = cpp_extension.CppExtension(
        name="pyFastFss" + "._C",
        sources=[
            # python
            *here_glob("pyFastFss", "csrc", "**", "*.cpp"),
            # public api
            *repo_glob("src", "*.cpp"),
            # cpu
            *repo_glob("src", "cpu", "*.cpp"),
        ],
        include_dirs=[
            repo("include"),
            repo("third_party", "wideint", "include"),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
        },
    )
else:
    if sys.platform == "linux":
        cxx_flags += ["-DFAST_FSS_ENABLE_CUDA"]
    else:
        cxx_flags += ["/DFAST_FSS_ENABLE_CUDA"]
    ext_module = cpp_extension.CUDAExtension(
        name="pyFastFss" + "._C",
        sources=[
            # python
            *here_glob("pyFastFss", "csrc", "**", "*.cpp"),
            *here_glob("pyFastFss", "csrc", "**", "*.cu"),
            # public api
            *repo_glob("src", "*.cpp"),
            # cpu
            *repo_glob("src", "cpu", "*.cpp"),
            # cuda
            *repo_glob("src", "cuda", "*.cpp"),
            *repo_glob("src", "cuda", "*.cu"),
        ],
        include_dirs=[
            repo("include"),
            repo("third_party", "wideint", "include"),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )

setup(
    name="pyFastFss",
    version="0.0.202605040050",
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
