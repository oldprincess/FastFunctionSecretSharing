# Fast Function Secret Sharing

## How to build

Initialize submodules first:

```shell
git submodule update --init --recursive
```

### CMake

Configure and build the C++ library.
If a CUDA toolchain is available, CUDA sources are enabled automatically:

Windows:

```shell
cmake -S . -B build
cmake --build build --config Release
```

Linux:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Build and run tests explicitly:

Windows:

```shell
cmake -S . -B build -DFAST_FSS_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Linux:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAST_FSS_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

If CUDA is not available or you only want CPU targets:

Windows:

```shell
cmake -S . -B build -DFAST_FSS_ENABLE_CUDA=OFF
cmake --build build --config Release
```

Linux:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAST_FSS_ENABLE_CUDA=OFF
cmake --build build
```

Build CPU-only targets and tests:

Windows:

```shell
cmake -S . -B build -DFAST_FSS_ENABLE_CUDA=OFF -DFAST_FSS_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Linux:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFAST_FSS_ENABLE_CUDA=OFF -DFAST_FSS_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Use Python API

make sure you have installed

- nvcc (cuda compiler)
- python
- pytorch

run

```shell
cd lib/torch_extension
python setup.py bdist_wheel
pip install dist/pyFastFss-<version>.whl
```

## References

### OTTT (One-time truth tables)

> Ishai Y, Kushilevitz E, Meldgaard S, et al. On the power of correlated randomness in secure computation[C]//Theory of cryptography conference. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013: 600-620.
> https://www.iacr.org/archive/tcc2013/77850598/77850598.pdf

### DCF (Distributed Comparision Function)

> Boyle E, Chandran N, Gilboa N, et al. Function secret sharing for mixed-mode and fixed-point secure computation[C]//Annual International Conference on the Theory and Applications of Cryptographic Techniques. Cham: Springer International Publishing, 2021: 871-900.
> https://eprint.iacr.org/2020/1392.pdf

### DPF (Distributed Point Function)

> Boyle E, Gilboa N, Ishai Y. Function secret sharing: Improvements and extensions[C]//Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016: 1292-1303.
> https://eprint.iacr.org/2018/707.pdf

### Grotto

> Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC or Z2n via (2, 2)-DPFs[C]//Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security. 2023: 2143-2157.
> https://eprint.iacr.org/2023/108.pdf
