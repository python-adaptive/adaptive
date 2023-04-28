from typing import Union

import numpy as np

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    # Remove this when we drop support for Python 3.9
    from typing_extensions import TypeAlias

Float: TypeAlias = Union[float, np.float_]
Bool: TypeAlias = Union[bool, np.bool_]
Int: TypeAlias = Union[int, np.int_]
Real: TypeAlias = Union[Float, Int]


__all__ = ["Float", "Bool", "Int", "Real"]
