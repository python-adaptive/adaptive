import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from adaptive import AverageLearner1D
from adaptive.tests.test_learners import (
    generate_random_parametrization,
    noisy_peak,
    simple_run,
)


def almost_equal_dicts(a, b):
    assert_series_equal(pd.Series(sorted(a.items())), pd.Series(sorted(b.items())))


def test_tell_many_at_point():
    f = generate_random_parametrization(noisy_peak)
    learner = AverageLearner1D(f, bounds=[-2, 2])
    control = AverageLearner1D(f, bounds=[-2, 2])
    learner._recompute_losses_factor = 1
    control._recompute_losses_factor = 1
    simple_run(learner, 100)
    for x, samples in learner._data_samples.items():
        control.tell_many_at_point(x, samples)

    almost_equal_dicts(learner.data, control.data)
    almost_equal_dicts(learner.error, control.error)
    almost_equal_dicts(learner.rescaled_error, control.rescaled_error)
    almost_equal_dicts(learner.neighbors, control.neighbors)
    almost_equal_dicts(learner.neighbors_combined, control.neighbors_combined)
    assert learner.npoints == control.npoints
    assert learner.nsamples == control.nsamples
    assert len(learner._data_samples) == len(control._data_samples)
    assert learner._data_samples.keys() == control._data_samples.keys()

    for k, v1 in learner._data_samples.items():
        v2 = control._data_samples[k]
        assert len(v1) == len(v2)
        np.testing.assert_almost_equal(sorted(v1.values()), sorted(v2.values()))

    assert learner._bbox[0] == control._bbox[0]
    assert learner._bbox[1] == control._bbox[1]
    almost_equal_dicts(learner.losses, control.losses)
    np.testing.assert_almost_equal(learner.loss(), control.loss())

    # Try if the control is runnable
    simple_run(control, 200)
