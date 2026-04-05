from typing import TypeAlias

import numpy as np

Float: TypeAlias = float | np.float64
Bool: TypeAlias = bool | np.bool_
Int: TypeAlias = int | np.int_
Real: TypeAlias = Float | Int


__all__ = ["Float", "Bool", "Int", "Real"]
