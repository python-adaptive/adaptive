import sys
from typing import Union

import numpy as np

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

Float: TypeAlias = Union[float, np.float64]
Bool: TypeAlias = Union[bool, np.bool_]
Int: TypeAlias = Union[int, np.int_]
Real: TypeAlias = Union[Float, Int]


__all__ = ["Float", "Bool", "Int", "Real"]
