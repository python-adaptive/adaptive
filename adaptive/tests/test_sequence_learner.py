import asyncio

from adaptive import Runner, SequenceLearner
from adaptive.runner import SequentialExecutor


class FailOnce:
    def __init__(self):
        self.failed = False

    def __call__(self, value):
        if self.failed:
            return value
        self.failed = True
        raise RuntimeError


def test_fail_with_sequence_of_unhashable():
    # https://github.com/python-adaptive/adaptive/issues/265
    seq = [{1: 1}]  # unhashable
    learner = SequenceLearner(FailOnce(), sequence=seq)
    runner = Runner(
        learner, goal=SequenceLearner.done, retries=1, executor=SequentialExecutor()
    )
    asyncio.get_event_loop().run_until_complete(runner.task)
    assert runner.status() == "finished"
