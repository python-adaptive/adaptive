from numbers import Integral as Int
from numbers import Real
from typing import Union

import numpy as np

try:
    from typing import TypeAlias
except ImportError:
    # Remove this when we drop support for Python 3.9
    from typing_extensions import TypeAlias

Float: TypeAlias = Union[float, np.float_]
Bool: TypeAlias = Union[bool, np.bool_]


__all__ = ["Float", "Bool", "Int", "Real"]
