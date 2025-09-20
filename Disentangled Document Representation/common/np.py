# coding: utf-8
from config.config import GPU


if GPU:
    import cupy as np
    import cupyx  # scatter_add など cupy 拡張が必要な場合
else:
    import numpy as np