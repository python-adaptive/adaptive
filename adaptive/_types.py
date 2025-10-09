# Only used for static type checkers, should only be imported in `if TYPE_CHECKING` block
# Workaround described in https://github.com/agronholm/typeguard/issues/456

import concurrent.futures as concurrent
import sys
from typing import TYPE_CHECKING, TypeAlias

import distributed
import ipyparallel
import loky
import mpi4py.futures

from adaptive.utils import SequentialExecutor

# For Python 3.14+, include InterpreterPoolExecutor in the type alias
if sys.version_info >= (3, 14):
    if TYPE_CHECKING:
        # Type checkers will see this when checking Python 3.14+ code
        ExecutorTypes: TypeAlias = (
            concurrent.ProcessPoolExecutor
            | concurrent.ThreadPoolExecutor
            | concurrent.InterpreterPoolExecutor  # type: ignore[attr-defined]
            | SequentialExecutor
            | loky.reusable_executor._ReusablePoolExecutor
            | distributed.Client
            | distributed.cfexecutor.ClientExecutor
            | mpi4py.futures.MPIPoolExecutor
            | ipyparallel.Client
            | ipyparallel.client.view.ViewExecutor
        )
else:
    ExecutorTypes: TypeAlias = (
        concurrent.ProcessPoolExecutor
        | concurrent.ThreadPoolExecutor
        | SequentialExecutor
        | loky.reusable_executor._ReusablePoolExecutor
        | distributed.Client
        | distributed.cfexecutor.ClientExecutor
        | mpi4py.futures.MPIPoolExecutor
        | ipyparallel.Client
        | ipyparallel.client.view.ViewExecutor
    )
