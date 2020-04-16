import asyncio

import numpy as np

from adaptive import Runner, SequenceLearner
from adaptive.runner import SequentialExecutor


class FailOnce:
    def __init__(self):
        self.failed = False

    def __call__(self, value):
        if self.failed:
            return value
        self.failed = True
        raise Exception


def test_fail_with_sequence_of_unhashable():
    # https://github.com/python-adaptive/adaptive/issues/265
    seq = [dict(x=x) for x in np.linspace(-1, 1, 101)]  # unhashable
    learner = SequenceLearner(FailOnce(), sequence=seq)
    runner = Runner(
        learner, goal=SequenceLearner.done, retries=100, executor=SequentialExecutor()
    )  # with 100 retries the test will fail once in 10^31
    asyncio.get_event_loop().run_until_complete(runner.task)
    assert runner.status() == "finished"
