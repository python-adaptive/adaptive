import collections
import functools as ft
import inspect
import itertools as it
import math
import operator
import os
import random
import shutil
import tempfile
import time

import flaky
import numpy as np
import pytest
import scipy.spatial

import adaptive
from adaptive.learner import (
    AverageLearner,
    AverageLearner1D,
    BalancingLearner,
    DataSaver,
    IntegratorLearner,
    Learner1D,
    Learner2D,
    LearnerND,
    SequenceLearner,
)
from adaptive.learner.learner1D import with_pandas
from adaptive.runner import simple

try:
    from adaptive.learner.skopt_learner import SKOptLearner
except (ModuleNotFoundError, ImportError):
    # XXX: catch the ImportError because of https://github.com/scikit-optimize/scikit-optimize/issues/902
    SKOptLearner = None


LOSS_FUNCTIONS = {
    Learner1D: (
        "loss_per_interval",
        (
            adaptive.learner.learner1D.default_loss,
            adaptive.learner.learner1D.uniform_loss,
            adaptive.learner.learner1D.curvature_loss_function(),
        ),
    ),
    Learner2D: (
        "loss_per_triangle",
        (
            adaptive.learner.learner2D.default_loss,
            adaptive.learner.learner2D.uniform_loss,
            adaptive.learner.learner2D.minimize_triangle_surface_loss,
            adaptive.learner.learner2D.resolution_loss_function(),
        ),
    ),
    LearnerND: (
        "loss_per_simplex",
        (
            adaptive.learner.learnerND.default_loss,
            adaptive.learner.learnerND.std_loss,
            adaptive.learner.learnerND.uniform_loss,
        ),
    ),
}


def generate_random_parametrization(f):
    """Return a realization of 'f' with parameters bound to random values.

    Parameters
    ----------
    f : callable
        All parameters but the first must be annotated with a callable
        that, when called with no arguments, produces a value of the
        appropriate type for the parameter in question.
    """
    _, *params = inspect.signature(f).parameters.items()
    if any(not callable(v.annotation) for (p, v) in params):
        raise TypeError(
            f"All parameters to {f.__name__} must be annotated with functions."
        )
    realization = {p: v.annotation() for (p, v) in params}
    return ft.partial(f, **realization)


def uniform(a, b):
    return lambda: random.uniform(a, b)


def simple_run(learner, n):
    def get_goal(learner):
        if hasattr(learner, "nsamples"):
            return lambda l: l.nsamples > n
        else:
            return lambda l: l.npoints > n

    def goal():
        if isinstance(learner, BalancingLearner):
            return get_goal(learner.learners[0])
        elif isinstance(learner, DataSaver):
            return get_goal(learner.learner)
        return get_goal(learner)

    simple(learner, goal())


# Library of functions and associated learners.

learner_function_combos = collections.defaultdict(list)


def learn_with(learner_type, **init_kwargs):
    def _(f):
        learner_function_combos[learner_type].append((f, init_kwargs))
        return f

    return _


def xfail(learner):
    return pytest.mark.xfail, learner


def maybe_skip(learner):
    return (pytest.mark.skip, learner) if learner is None else learner


# All parameters except the first must be annotated with a callable that
# returns a random value for that parameter.


@learn_with(Learner1D, bounds=(-1, 1))
def quadratic(x, m: uniform(1, 4), b: uniform(0, 1)):
    return m * x**2 + b


@learn_with(Learner1D, bounds=(-1, 1))
@learn_with(SequenceLearner, sequence=np.linspace(-1, 1, 201))
def linear_with_peak(x, d: uniform(-1, 1)):
    a = 0.01
    return x + a**2 / (a**2 + (x - d) ** 2)


@learn_with(LearnerND, bounds=((-1, 1), (-1, 1)))
@learn_with(Learner2D, bounds=((-1, 1), (-1, 1)))
@learn_with(SequenceLearner, sequence=np.random.rand(1000, 2))
def ring_of_fire(xy, d: uniform(0.2, 1)):
    a = 0.2
    x, y = xy
    return x + math.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


@learn_with(LearnerND, bounds=((-1, 1), (-1, 1), (-1, 1)))
@learn_with(SequenceLearner, sequence=np.random.rand(1000, 3))
def sphere_of_fire(xyz, d: uniform(0.2, 0.5)):
    a = 0.2
    x, y, z = xyz
    return x + math.exp(-((x**2 + y**2 + z**2 - d**2) ** 2) / a**4) + z**2


@learn_with(SequenceLearner, sequence=range(1000))
@learn_with(AverageLearner, rtol=1)
def gaussian(n):
    return random.gauss(1, 1)


@learn_with(AverageLearner1D, bounds=(-2, 2))
def noisy_peak(
    seed_x,
    sigma: uniform(1.5, 2.5),
    peak_width: uniform(0.04, 0.06),
    offset: uniform(-0.6, -0.3),
):
    seed, x = seed_x
    y = x**3 - x + 3 * peak_width**2 / (peak_width**2 + (x - offset) ** 2)
    noise = np.random.normal(0, sigma)
    return y + noise


# Decorators for tests.


# Create a sequence of learner parameters by adding all
# possible loss functions to an existing parameter set.
def add_loss_to_params(learner_type, existing_params):
    if learner_type not in LOSS_FUNCTIONS:
        return [existing_params]
    loss_param, loss_functions = LOSS_FUNCTIONS[learner_type]
    loss_params = [{loss_param: f} for f in loss_functions]
    return [dict(**existing_params, **lp) for lp in loss_params]


def run_with(*learner_types, with_all_loss_functions=True):
    pars = []
    for learner in learner_types:
        has_marker = isinstance(learner, tuple)
        if has_marker:
            marker, learner = learner
        for f, k in learner_function_combos[learner]:
            ks = add_loss_to_params(learner, k) if with_all_loss_functions else [k]
            for k in ks:
                # Check if learner was marked with our `xfail` decorator
                # XXX: doesn't work when feeding kwargs to xfail.
                if has_marker:
                    pars.append(pytest.param(learner, f, dict(k), marks=[marker]))
                else:
                    pars.append((learner, f, dict(k)))
    return pytest.mark.parametrize("learner_type, f, learner_kwargs", pars)


def ask_randomly(learner, rounds, points):
    n_rounds = random.randrange(*rounds)
    n_points = [random.randrange(*points) for _ in range(n_rounds)]

    xs = []
    losses = []
    for n in n_points:
        new_xs, new_losses = learner.ask(n)
        xs.extend(new_xs)
        losses.extend(new_losses)

    return xs, losses


# Tests


@run_with(Learner1D)
def test_uniform_sampling1D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = ask_randomly(learner, (10, 20), (10, 20))

    points.sort()
    ivals = np.diff(sorted(points))
    assert max(ivals) / min(ivals) < 2 + 1e-8


@pytest.mark.xfail
@run_with(Learner2D, LearnerND)
def test_uniform_sampling2D(learner_type, f, learner_kwargs):
    """Points are sampled uniformly if no data is provided.

    Non-uniform sampling implies that we think we know something about
    the function, which we do not in the absence of data.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)

    points, _ = ask_randomly(learner, (70, 100), (10, 20))
    tree = scipy.spatial.cKDTree(points)

    # regular grid
    n = math.sqrt(len(points))
    xbounds, ybounds = learner_kwargs["bounds"]
    r = math.sqrt((ybounds[1] - ybounds[0]) / (xbounds[1] - xbounds[0]))
    xs, dx = np.linspace(*xbounds, int(n / r), retstep=True)
    ys, dy = np.linspace(*ybounds, int(n * r), retstep=True)

    distances, neighbors = tree.query(list(it.product(xs, ys)), k=1)
    assert max(distances) < math.sqrt(dx**2 + dy**2)


@pytest.mark.parametrize(
    "learner_type, bounds",
    [
        (Learner1D, (-1, 1)),
        (Learner2D, ((-1, 1), (-1, 1))),
        (LearnerND, ((-1, 1), (-1, 1), (-1, 1))),
    ],
)
def test_learner_accepts_lists(learner_type, bounds):
    def f(x):
        return [0, 1]

    learner = learner_type(f, bounds=bounds)
    simple_run(learner, 10)


@run_with(Learner1D, Learner2D, LearnerND, SequenceLearner, AverageLearner1D)
def test_adding_existing_data_is_idempotent(learner_type, f, learner_kwargs):
    """Adding already existing data is an idempotent operation.

    Either it is idempotent, or it is an error.
    This is the only sane behaviour.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner.new()
    if learner_type in (Learner1D, AverageLearner1D):
        learner._recompute_losses_factor = 1
        control._recompute_losses_factor = 1

    N = random.randint(10, 30)
    control.ask(N)
    xs, _ = learner.ask(N)
    points = [(x, learner.function(x)) for x in xs]

    for p in points:
        control.tell(*p)
        learner.tell(*p)

    random.shuffle(points)
    for p in points:
        learner.tell(*p)

    M = random.randint(10, 30)
    pls = zip(*learner.ask(M))
    cpls = zip(*control.ask(M))
    if learner_type is SequenceLearner:
        # The SequenceLearner's points might not be hasable
        points, values = zip(*pls)
        indices, points = zip(*points)

        cpoints, cvalues = zip(*cpls)
        cindices, cpoints = zip(*cpoints)
        assert (np.array(points) == np.array(cpoints)).all()
        assert values == cvalues
        assert indices == cindices
    else:
        # Point ordering is not defined, so compare as sets
        assert set(pls) == set(cpls)


# XXX: This *should* pass (https://github.com/python-adaptive/adaptive/issues/55)
#      but we xfail it now, as Learner2D will be deprecated anyway
@run_with(
    Learner1D,
    xfail(Learner2D),
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    SequenceLearner,
)
def test_adding_non_chosen_data(learner_type, f, learner_kwargs):
    """Adding data for a point that was not returned by 'ask'."""
    # XXX: learner, control and bounds are not defined
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner.new()

    if learner_type is Learner2D:
        # If the stack_size is bigger then the number of points added,
        # ask will return a point from the _stack.
        learner.stack_size = 1
        control.stack_size = 1

    N = random.randint(10, 30)
    xs, _ = control.ask(N)

    ys = [learner.function(x) for x in xs]
    for x, y in zip(xs, ys):
        control.tell(x, y)
        learner.tell(x, y)

    M = random.randint(10, 30)
    pls = zip(*learner.ask(M))
    cpls = zip(*control.ask(M))

    if learner_type is SequenceLearner:
        # The SequenceLearner's points might not be hasable
        points, values = zip(*pls)
        indices, points = zip(*points)

        cpoints, cvalues = zip(*cpls)
        cindices, cpoints = zip(*cpoints)
        assert (np.array(points) == np.array(cpoints)).all()
        assert values == cvalues
        assert indices == cindices
    else:
        # Point ordering within a single call to 'ask'
        # is not guaranteed to be the same by the API.
        assert set(pls) == set(cpls)


@run_with(
    Learner1D, xfail(Learner2D), xfail(LearnerND), AverageLearner, AverageLearner1D
)
def test_point_adding_order_is_irrelevant(learner_type, f, learner_kwargs):
    """The order of calls to 'tell' between calls to 'ask'
    is arbitrary.

    This test will fail for the Learner2D because
    `interpolate.interpnd.estimate_gradients_2d_global` will give different
    outputs based on the order of the triangles and values in
    (ip.tri, ip.values). Therefore the _stack will contain different points.
    """
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner.new()

    if learner_type in (Learner1D, AverageLearner1D):
        learner._recompute_losses_factor = 1
        control._recompute_losses_factor = 1

    N = random.randint(10, 30)
    control.ask(N)
    xs, _ = learner.ask(N)
    points = [(x, learner.function(x)) for x in xs]

    for p in points:
        control.tell(*p)

    random.shuffle(points)
    for p in points:
        learner.tell(*p)

    M = random.randint(10, 30)
    pls = sorted(zip(*learner.ask(M)))
    cpls = sorted(zip(*control.ask(M)))
    # Point ordering within a single call to 'ask'
    # is not guaranteed to be the same by the API.
    # We compare the sorted points instead of set, because the points
    # should only be identical up to machine precision.
    if isinstance(pls[0][0], tuple):
        # This is the case for AverageLearner1D
        pls = [(*x, y) for x, y in pls]
        cpls = [(*x, y) for x, y in cpls]
    np.testing.assert_almost_equal(pls, cpls)


# XXX: the Learner2D fails with ~50% chance
# see https://github.com/python-adaptive/adaptive/issues/55
@run_with(Learner1D, xfail(Learner2D), LearnerND, AverageLearner, AverageLearner1D)
def test_expected_loss_improvement_is_less_than_total_loss(
    learner_type, f, learner_kwargs
):
    """The estimated loss improvement can never be greater than the total loss."""
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    for _ in range(2):
        # We do this twice to make sure that the AverageLearner1D
        # has two different points in `x`.
        N = random.randint(50, 100)
        xs, loss_improvements = learner.ask(N)
        for x in xs:
            learner.tell(x, learner.function(x))

    M = random.randint(50, 100)
    _, loss_improvements = learner.ask(M)

    if learner_type is Learner2D:
        assert sum(loss_improvements) < sum(
            learner.loss_per_triangle(learner.interpolator(scaled=True))
        )
    elif learner_type in (Learner1D, AverageLearner1D):
        assert sum(loss_improvements) < sum(learner.losses.values())
    elif learner_type is AverageLearner:
        assert sum(loss_improvements) < learner.loss()


# XXX: This *should* pass (https://github.com/python-adaptive/adaptive/issues/55)
#      but we xfail it now, as Learner2D will be deprecated anyway
@run_with(Learner1D, xfail(Learner2D), LearnerND, AverageLearner1D)
def test_learner_performance_is_invariant_under_scaling(
    learner_type, f, learner_kwargs
):
    """Learners behave identically under transformations that leave
       the loss invariant.

    This is a statement that the learner makes decisions based solely
    on the loss function.
    """
    # for now we just scale X and Y by random factors
    f = generate_random_parametrization(f)
    if learner_type is AverageLearner1D:
        # no noise for AverageLearner1D to make it deterministic
        f = ft.partial(f, sigma=0)

    control_kwargs = dict(learner_kwargs)
    control = learner_type(f, **control_kwargs)

    xscale = 1000 * random.random()
    yscale = 1000 * random.random()

    l_kwargs = dict(learner_kwargs)
    bounds = xscale * np.array(l_kwargs["bounds"])
    bounds = tuple((bounds).tolist())  # to satisfy typeguard tests
    l_kwargs["bounds"] = bounds

    def scale_x(x):
        if isinstance(learner, AverageLearner1D):
            seed, x = x
            return (seed, x / xscale)
        return np.array(x) / xscale

    learner = learner_type(lambda x: yscale * f(scale_x(x)), **l_kwargs)

    if learner_type in [Learner1D, LearnerND, AverageLearner1D]:
        learner._recompute_losses_factor = 1
        control._recompute_losses_factor = 1

    npoints = random.randrange(300, 500)

    if learner_type is LearnerND:
        # Because the LearnerND is slow
        npoints //= 10

    for n in range(npoints):
        cxs, _ = control.ask(1)
        xs, _ = learner.ask(1)
        control.tell_many(cxs, [control.function(x) for x in cxs])
        learner.tell_many(xs, [learner.function(x) for x in xs])

        # Check whether the points returned are the same
        xs_unscaled = [scale_x(x) for x in xs]
        assert np.allclose(xs_unscaled, cxs)

    # Check if the losses are close
    assert math.isclose(learner.loss(), control.loss(), rel_tol=1e-10)


@flaky.flaky(max_runs=10)
@run_with(
    Learner1D,
    Learner2D,
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    SequenceLearner,
    with_all_loss_functions=False,
)
def test_balancing_learner(learner_type, f, learner_kwargs):
    """Test if the BalancingLearner works with the different types of learners."""
    learners = [
        learner_type(generate_random_parametrization(f), **learner_kwargs)
        for i in range(4)
    ]

    learner = BalancingLearner(learners)

    # Emulate parallel execution
    stash = []

    for i in range(100):
        n = random.randint(1, 10)
        m = random.randint(0, n)
        xs, _ = learner.ask(n, tell_pending=False)

        # Save 'm' random points out of `xs` for later
        random.shuffle(xs)
        for _ in range(m):
            stash.append(xs.pop())

        for x in xs:
            learner.tell(x, learner.function(x))

        # Evaluate and add 'm' random points from `stash`
        random.shuffle(stash)
        for _ in range(m):
            x = stash.pop()
            learner.tell(x, learner.function(x))

    if learner_type is AverageLearner1D:
        nsamples = [l.nsamples for l in learner.learners]
        assert all(l.nsamples > 5 for l in learner.learners), nsamples
    else:
        npoints = [l.npoints for l in learner.learners]
        assert all(l.npoints > 5 for l in learner.learners), npoints


@run_with(
    Learner1D,
    Learner2D,
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    maybe_skip(SKOptLearner),
    IntegratorLearner,
    SequenceLearner,
    with_all_loss_functions=False,
)
def test_saving(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    learner = learner_type(f, **learner_kwargs)
    control = learner.new()
    if learner_type in (Learner1D, AverageLearner1D):
        learner._recompute_losses_factor = 1
        control._recompute_losses_factor = 1
    simple_run(learner, 100)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        learner.save(path)
        control.load(path)

        np.testing.assert_almost_equal(learner.loss(), control.loss())

        # Try if the control is runnable
        simple_run(control, 200)
    finally:
        os.remove(path)


@run_with(
    Learner1D,
    Learner2D,
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    maybe_skip(SKOptLearner),
    IntegratorLearner,
    SequenceLearner,
    with_all_loss_functions=False,
)
def test_saving_of_balancing_learner(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    learner = BalancingLearner([learner_type(f, **learner_kwargs)])
    control = learner.new()

    if learner_type in (Learner1D, AverageLearner1D):
        for l, c in zip(learner.learners, control.learners):
            l._recompute_losses_factor = 1
            c._recompute_losses_factor = 1

    simple_run(learner, 100)
    folder = tempfile.mkdtemp()

    def fname(learner):
        return folder + "test"

    try:
        learner.save(fname=fname)
        control.load(fname=fname)

        np.testing.assert_almost_equal(learner.loss(), control.loss())

        # Try if the control is runnable
        simple_run(control, 200)
    finally:
        shutil.rmtree(folder)


@run_with(
    Learner1D,
    Learner2D,
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    maybe_skip(SKOptLearner),
    IntegratorLearner,
    with_all_loss_functions=False,
)
def test_saving_with_datasaver(learner_type, f, learner_kwargs):
    f = generate_random_parametrization(f)
    g = lambda x: {"y": f(x), "t": random.random()}  # noqa: E731
    arg_picker = operator.itemgetter("y")
    learner = DataSaver(learner_type(g, **learner_kwargs), arg_picker)
    control = learner.new()

    if learner_type in (Learner1D, AverageLearner1D):
        learner.learner._recompute_losses_factor = 1
        control.learner._recompute_losses_factor = 1

    simple_run(learner, 100)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        learner.save(path)
        control.load(path)

        np.testing.assert_almost_equal(learner.loss(), control.loss())

        assert learner.extra_data == control.extra_data

        # Try if the control is runnable
        simple_run(learner, 200)
    finally:
        os.remove(path)


@pytest.mark.xfail
@run_with(Learner1D, Learner2D, LearnerND)
def test_convergence_for_arbitrary_ordering(learner_type, f, learner_kwargs):
    """Learners that are learning the same function should converge
    to the same result "eventually" if given the same data, regardless
    of the order in which that data is given.
    """
    # XXX: not sure how to implement this. Can we say anything at all about
    #      the scaling of the loss with the number of points?
    raise NotImplementedError()


@pytest.mark.xfail
@run_with(Learner1D, Learner2D, LearnerND)
def test_learner_subdomain(learner_type, f, learner_kwargs):
    """Learners that never receive data outside of a subdomain should
    perform 'similarly' to learners defined on that subdomain only."""
    # XXX: not sure how to implement this. How do we measure "performance"?
    raise NotImplementedError()


def add_time(f):
    @ft.wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        return {"result": result, "time": time.time() - t0}

    return wrapper


@pytest.mark.skipif(not with_pandas, reason="pandas is not installed")
@run_with(
    Learner1D,
    Learner2D,
    LearnerND,
    AverageLearner,
    AverageLearner1D,
    SequenceLearner,
    IntegratorLearner,
    with_all_loss_functions=False,
)
def test_to_dataframe(learner_type, f, learner_kwargs):
    import pandas

    if learner_type is LearnerND:
        kw = {"point_names": tuple("xyz")[: len(learner_kwargs["bounds"])]}
    else:
        kw = {}

    learner = learner_type(generate_random_parametrization(f), **learner_kwargs)

    # Test empty dataframe
    df = learner.to_dataframe(**kw)
    assert len(df) == 0
    assert "inputs" in df.attrs
    assert "output" in df.attrs

    # Run the learner
    simple_run(learner, 100)
    df = learner.to_dataframe(**kw)
    assert isinstance(df, pandas.DataFrame)
    if learner_type is AverageLearner1D:
        assert len(df) == learner.nsamples
    else:
        assert len(df) == learner.npoints

    # Add points from the DataFrame to a new empty learner
    learner2 = learner.new()
    learner2.load_dataframe(df, **kw)
    assert learner2.npoints == learner.npoints

    # Test this for a learner in a BalancingLearner
    learners = [
        learner_type(generate_random_parametrization(f), **learner_kwargs)
        for _ in range(2)
    ]
    bal_learner = BalancingLearner(learners)
    simple_run(bal_learner, 100)
    df_bal = bal_learner.to_dataframe(**kw)
    assert isinstance(df_bal, pandas.DataFrame)

    if learner_type is not AverageLearner1D:
        assert len(df_bal) == bal_learner.npoints

    # Test loading from a DataFrame into the BalancingLearner
    learners2 = [
        learner_type(generate_random_parametrization(f), **learner_kwargs)
        for _ in range(2)
    ]
    bal_learner2 = BalancingLearner(learners2)
    bal_learner2.load_dataframe(df_bal, **kw)
    assert bal_learner2.npoints == bal_learner.npoints

    if learner_type is SequenceLearner:
        # We do not test the DataSaver with the SequenceLearner
        # because the DataSaver is not compatible with the SequenceLearner.
        return

    # Test with DataSaver
    learner = learner_type(
        add_time(generate_random_parametrization(f)), **learner_kwargs
    )
    data_saver = DataSaver(learner, operator.itemgetter("result"))
    df = data_saver.to_dataframe(**kw)  # test if empty dataframe works
    simple_run(data_saver, 100)
    df = data_saver.to_dataframe(**kw)
    if learner_type is AverageLearner1D:
        assert len(df) == data_saver.nsamples
    else:
        assert len(df) == data_saver.npoints

    # Test loading from a DataFrame into a new DataSaver
    data_saver2 = data_saver.new()
    data_saver2.load_dataframe(df, **kw)
    assert data_saver2.extra_data.keys() == data_saver.extra_data.keys()
    assert all(
        data_saver2.extra_data[k] == data_saver.extra_data[k]
        for k in data_saver.extra_data.keys()
    )
