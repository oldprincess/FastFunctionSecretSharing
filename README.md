# Fast Function Secret Sharing

## How to build

### Use Python API(Recommended)

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

### Use C++ API

```shell
# Unix
nvcc -I include/ src/cpu/*.cpp src/cuda/*.cu src/cuda/*.cpp -shared -Xcompiler "-fPIC" -Xcompiler "-maes" -Xcompiler "-fopenmp" -o libFastFss.so -std=c++17 --expt-relaxed-constexpr
```

## References

### DCF

> Boyle E, Chandran N, Gilboa N, et al. Function secret sharing for mixed-mode and fixed-point secure computation[C]//Annual International Conference on the Theory and Applications of Cryptographic Techniques. Cham: Springer International Publishing, 2021: 871-900.

### DPF(Distributed Point Function)

> Boyle E, Gilboa N, Ishai Y. Function secret sharing: Improvements and extensions[C]//Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016: 1292-1303.

### Grotto

> Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC or Z2n via (2, 2)-DPFs[C]//Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security. 2023: 2143-2157.
