# Only used for static type checkers, should only be imported in `if TYPE_CHECKING` block
# Workaround described in https://github.com/agronholm/typeguard/issues/456

import concurrent.futures as concurrent
from typing import TypeAlias

import distributed
import ipyparallel
import loky
import mpi4py.futures

from adaptive.utils import SequentialExecutor

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
