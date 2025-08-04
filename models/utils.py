# uncompyle6 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Nov 26 2021, 20:14:08) 
# [GCC 9.3.0]
# Embedded file name: /data/home/gcc/Fusiformer_new/models/utils.py
# Compiled at: 2025-02-27 06:02:12
# Size of source mod 2**32: 1399 bytes
from itertools import repeat
import collections.abc

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def drop_path(x, drop_prob: float=0.0, training: bool=False, scale_by_keep: bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    return drop_prob == 0.0 or training or x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1, ) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        if scale_by_keep:
            random_tensor.div_(keep_prob)
    return x * random_tensor

# okay decompiling utils.cpython-38.pyc
