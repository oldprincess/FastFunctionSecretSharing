# 使用pytorch调用FastFss

## 1. 环境配置

Python

pytorch(需要支持CUDA)

## 2. 安装FastFss

生成wheel文件

```bash
python setup.py bdist_wheel
```

安装wheel文件

```bash
# python3.9, windows, amd64
pip install dist/pyFastFss-0.0.1-cp39-cp39-win_amd64.whl
# python3.10 linux, x86_64
pip install dist/pyFastFss-0.0.1-cp310-cp310-linux_x86_64.whl
```

```text
优化1: CUDA紧凑AES加密, 减少存储轮密钥的开销
注意1: 查找表需要存在global memory，充分利用L1缓存
能用32位尽量用32位, 32位能减少寄存器的使用
```
