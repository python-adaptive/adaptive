import asyncio

import numpy as np

import adaptive


def might_fail(dct):
    import random

    if random.random() < 0.5:
        raise Exception()
    return dct["x"]


def test_fail_with_sequence_of_unhashable():
    # https://github.com/python-adaptive/adaptive/issues/265
    seq = [dict(x=x) for x in np.linspace(-1, 1, 101)]  # unhashable
    learner = adaptive.SequenceLearner(might_fail, sequence=seq)
    runner = adaptive.Runner(
        learner, goal=adaptive.SequenceLearner.done, retries=100
    )  # with 100 retries the test will fail once in 10^31
    asyncio.get_event_loop().run_until_complete(runner.task)
    assert runner.status() == "finished"
