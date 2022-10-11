from typing import Union

import numpy as np

try:
    from typing import TypeAlias
except ImportError:
    # Remove this when we drop support for Python 3.9
    from typing_extensions import TypeAlias

Float: TypeAlias = Union[float, np.float_]
Int: TypeAlias = Union[int, np.int_]
Real: TypeAlias = Union[Float, Int]
