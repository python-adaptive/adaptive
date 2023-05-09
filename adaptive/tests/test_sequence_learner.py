import asyncio

import pytest

from adaptive import Runner, SequenceLearner
from adaptive.learner.learner1D import with_pandas
from adaptive.runner import SequentialExecutor, simple

offset = 0.0123


def peak(x, offset=offset, wait=True):
    a = 0.01
    return {"x": x + a**2 / (a**2 + (x - offset) ** 2)}


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
    runner = Runner(learner, retries=1, executor=SequentialExecutor())
    asyncio.get_event_loop().run_until_complete(runner.task)
    assert runner.status() == "finished"


@pytest.mark.skipif(not with_pandas, reason="pandas is not installed")
def test_save_load_dataframe():
    learner = SequenceLearner(peak, sequence=range(10, 30, 1))
    simple(learner, npoints_goal=10)
    df = learner.to_dataframe()
    assert len(df) == 10
    assert df["x"].iloc[0] == 10
    df_full = learner.to_dataframe(full_sequence=True)
    assert len(df_full) == 20
    assert df_full["x"].iloc[0] == 10
    assert df_full["x"].iloc[-1] == 29

    learner2 = learner.new()
    assert learner2.data == {}
    learner2.load_dataframe(df)
    assert len(learner2.data) == 10
    assert learner.to_dataframe().equals(df)

    learner3 = learner.new()
    learner3.load_dataframe(df_full, full_sequence=True)
    assert len(learner3.data) == 10
    assert learner3.to_dataframe(full_sequence=True).equals(df_full)
