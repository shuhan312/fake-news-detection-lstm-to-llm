"""
通用工具函数 / Shared utilities.

目前只有 set_seed：在每个 notebook 开头调用一次，让实验结果可复现。
Currently provides set_seed: call it once at the top of every notebook so
results are reproducible run-to-run.
"""

import os
import random

import numpy as np

# 全项目统一使用的默认随机种子 / Project-wide default seed
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic_tf: bool = False) -> None:
    """固定 Python / NumPy / TensorFlow / PyTorch 的随机种子。
    Fix the random seeds of Python, NumPy, TensorFlow and PyTorch.

    TensorFlow 和 PyTorch 只在已安装时才设置，所以没装深度学习库的环境
    （例如只跑泄漏审计）也能正常调用。
    TensorFlow / PyTorch are only seeded if installed, so this also works in
    environments that only run the leakage audit.

    Args:
        seed: 随机种子 / the seed value.
        deterministic_tf: 为 True 时开启 TensorFlow 的确定性算子。结果完全可复现，
            但训练会变慢；需要逐位一致的结果时再打开。
            If True, enable TensorFlow op determinism. Fully reproducible but
            slower — turn on only when bit-identical results are required.

    注意 / Note:
        即使固定了种子，GPU / Apple MPS 上的部分算子仍可能有微小的不确定性。
        所以比较模型时仍应跑多个种子并报告 均值 ± 标准差，而不是只看一次的结果。
        Even with fixed seeds, some GPU / Apple MPS kernels are not bit-exact.
        Compare models across several seeds (mean ± std), never a single run.
    """
    # PYTHONHASHSEED 影响 Python 内置 hash 的随机化（例如 set 的遍历顺序）
    # PYTHONHASHSEED controls hash randomisation (e.g. set iteration order)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic_tf:
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
