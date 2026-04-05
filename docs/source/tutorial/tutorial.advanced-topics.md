---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  name: python3
---
(TutorialAdvancedTopics)=
# Advanced Topics

```{note}
Because this documentation consists of static html, the `live_plot` and `live_info` widget is not live.
Download the notebook in order to see the real behaviour. [^download]
```

```{code-cell} ipython3
:tags: [hide-cell]

import adaptive

adaptive.notebook_extension()

import asyncio
import random

offset = random.uniform(-0.5, 0.5)


def f(x, offset=offset):
    a = 0.01
    return x + a**2 / (a**2 + (x - offset) ** 2)
```

## Saving and loading learners

Every learner has a {class}`~adaptive.BaseLearner.save` and {class}`~adaptive.BaseLearner.load` method that can be used to save and load **only** the data of a learner.

Use the `fname` argument in `learner.save(fname=...)`.

Or, when using a {class}`~adaptive.BalancingLearner` one can use either a callable that takes the child learner and returns a filename **or** a list of filenames.

By default the resulting pickle files are compressed, to turn this off use `learner.save(fname=..., compress=False)`

```{code-cell} ipython3
# Let's create two learners and run only one.
learner = adaptive.Learner1D(f, bounds=(-1, 1))
control = adaptive.Learner1D(f, bounds=(-1, 1))

# Let's only run the learner
runner = adaptive.Runner(learner, loss_goal=0.01)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
fname = "data/example_file.p"
learner.save(fname)
control.load(fname)

(learner.plot().relabel("saved learner") + control.plot().relabel("loaded learner"))
```

Or just (without saving):

```{code-cell} ipython3
control = adaptive.Learner1D(f, bounds=(-1, 1))
control.copy_from(learner)
```

One can also periodically save the learner while running in a {class}`~adaptive.Runner`. Use it like:

```{code-cell} ipython3
def slow_f(x):
    from time import sleep

    sleep(5)
    return x


learner = adaptive.Learner1D(slow_f, bounds=[0, 1])
runner = adaptive.Runner(learner, npoints_goal=100)
runner.start_periodic_saving(
    save_kwargs={"fname": "data/periodic_example.p"}, interval=6
)
```

```{code-cell} ipython3
:tags: [hide-cell]

await asyncio.sleep(6)  # This is not needed in a notebook environment!
runner.cancel()
```

```{code-cell} ipython3
runner.live_info()  # we cancelled it after 6 seconds
```

```{code-cell} ipython3
# See the data 6 later seconds with
#!ls -lah data  # only works on macOS and Linux systems
```

## A watched pot never boils!

The {class}`adaptive.Runner` does its work in an `asyncio` task that runs concurrently with the IPython kernel, when using `adaptive` from a Jupyter notebook.
This is advantageous because it allows us to do things like live-updating plots, however it can trip you up if you’re not careful.

Notably: **if you block the IPython kernel, the runner will not do any work**.

For example if you wanted to wait for a runner to complete, **do not wait in a busy loop**:

```python
while not runner.task.done():
    pass
```

If you do this then **the runner will never finish**.

What to do if you don’t care about live plotting, and just want to run something until its done?

The simplest way to accomplish this is to use {class}`adaptive.BlockingRunner`:

```{code-cell} ipython3
learner = adaptive.Learner1D(f, bounds=(-1, 1))
adaptive.BlockingRunner(learner, loss_goal=0.01)
# This will only get run after the runner has finished
learner.plot()
```

## Reproducibility

By default `adaptive` runners evaluate the learned function in parallel across several cores.
The runners are also opportunistic, in that as soon as a result is available they will feed it to the learner and request another point to replace the one that just finished.

Because the order in which computations complete is non-deterministic, this means that the runner behaves in a non-deterministic way.
Adaptive makes this choice because in many cases the speedup from parallel execution is worth sacrificing the “purity” of exactly reproducible computations.

Nevertheless it is still possible to run a learner in a deterministic way with adaptive.

The simplest way is to use {class}`adaptive.runner.simple` to run your learner:

```{code-cell} ipython3
learner = adaptive.Learner1D(f, bounds=(-1, 1))

# blocks until completion
adaptive.runner.simple(learner, loss_goal=0.01)

learner.plot()
```

Note that unlike {class}`adaptive.Runner`, {class}`adaptive.runner.simple` *blocks* until it is finished.

If you want to enable determinism, want to continue using the non-blocking {class}`adaptive.Runner`, you can use the {class}`adaptive.runner.SequentialExecutor`:

```{code-cell} ipython3
from adaptive.runner import SequentialExecutor

learner = adaptive.Learner1D(f, bounds=(-1, 1))
runner = adaptive.Runner(learner, executor=SequentialExecutor(), loss_goal=0.01)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot(update_interval=0.1)
```

## Cancelling a runner

Sometimes you want to interactively explore a parameter space, and want the function to be evaluated at finer and finer resolution and manually control when the calculation stops.

If no `goal` is provided to a runner then the runner will run until cancelled.

`runner.live_info()` will provide a button that can be clicked to stop the runner.
You can also stop the runner programatically using `runner.cancel()`.

```{code-cell} ipython3
learner = adaptive.Learner1D(f, bounds=(-1, 1))
runner = adaptive.Runner(learner)
```

```{code-cell} ipython3
:tags: [hide-cell]

await asyncio.sleep(0.1)  # This is not needed in the notebook!
```

```{code-cell} ipython3
runner.cancel()  # Let's execute this after 0.1 seconds
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot(update_interval=0.1)
```

```{code-cell} ipython3
print(runner.status())
```

## Debugging Problems

Runners work in the background with respect to the IPython kernel, which makes it convenient, but also means that inspecting errors is more difficult because exceptions will not be raised directly in the notebook.
Often the only indication you will have that something has gone wrong is that nothing will be happening.

Let’s look at the following example, where the function to be learned will raise an exception 10% of the time.

```{code-cell} ipython3
def will_raise(x):
    from random import random
    from time import sleep

    sleep(random())
    if random() < 0.1:
        raise RuntimeError("something went wrong!")
    return x**2


learner = adaptive.Learner1D(will_raise, (-1, 1))
runner = adaptive.Runner(
    learner
)  # without 'goal' the runner will run forever unless cancelled
```

```{code-cell} ipython3
:tags: [hide-cell]

await asyncio.sleep(4)  # in 4 seconds it will surely have failed
```

```{code-cell} ipython3
runner.live_info()
```

```{code-cell} ipython3
runner.live_plot()
```

The above runner should continue forever, but we notice that it stops after a few points are evaluated.

First we should check that the runner has really finished:

```{code-cell} ipython3
runner.task.done()
```

If it has indeed finished then we should check the `result` of the runner.
This should be `None` if the runner stopped successfully.
If the runner stopped due to an exception then asking for the result will raise the exception with the stack trace:

```{code-cell} ipython3
:tags: [raises-exception]

runner.task.result()
```

You can also check `runner.tracebacks` which is a list of tuples with `(point, traceback)`.

```{code-cell} ipython3
for point, tb in runner.tracebacks:
    print(f"point: {point}:\n {tb}")
```

### Logging runners

Runners do their job in the background, which makes introspection quite cumbersome.
One way to inspect runners is to instantiate one with `log=True`:

```{code-cell} ipython3
learner = adaptive.Learner1D(f, bounds=(-1, 1))
runner = adaptive.Runner(learner, loss_goal=0.01, log=True)
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
runner.live_info()
```

This gives a the runner a `log` attribute, which is a list of the `learner` methods that were called, as well as their arguments.
This is useful because executors typically execute their tasks in a non-deterministic order.

This can be used with {class}`adaptive.runner.replay_log` to perfom the same set of operations on another runner:

```{code-cell} ipython3
reconstructed_learner = adaptive.Learner1D(f, bounds=learner.bounds)
adaptive.runner.replay_log(reconstructed_learner, runner.log)
```

```{code-cell} ipython3
learner.plot().Scatter.I.opts(size=6) * reconstructed_learner.plot()
```

## Adding coroutines

In the following example we'll add a {class}`~asyncio.Task` that times the runner.
This is *only* for demonstration purposes because one can simply check `runner.elapsed_time()` or use the `runner.live_info()` widget to see the time since the runner has started.

So let's get on with the example. To time the runner you **cannot** simply use

```python
now = datetime.now()
runner = adaptive.Runner(...)
print(datetime.now() - now)
```

because this will be done immediately. Also blocking the kernel with `while not runner.task.done()` will not work because the runner will not do anything when the kernel is blocked.

Therefore you need to create an `async` function and hook it into the `ioloop` like so:

```{code-cell} ipython3
import asyncio


async def time(runner):
    from datetime import datetime

    now = datetime.now()
    await runner.task
    return datetime.now() - now


ioloop = asyncio.get_event_loop()

learner = adaptive.Learner1D(f, bounds=(-1, 1))
runner = adaptive.Runner(learner, loss_goal=0.01)

timer = ioloop.create_task(time(runner))
```

```{code-cell} ipython3
:tags: [hide-cell]

await runner.task  # This is not needed in a notebook environment!
```

```{code-cell} ipython3
# The result will only be set when the runner is done.
timer.result()
```
(CustomParallelization)=
## Custom parallelization using coroutines

Adaptive by itself does not implement a way of sharing partial results between function executions.
Instead its implementation of parallel computation using executors is minimal by design.
The appropriate way to implement custom parallelization is by using coroutines (asynchronous functions).


We illustrate this approach by using `dask.distributed` for parallel computations in part because it supports asynchronous operation out-of-the-box.
We will focus on a function `f(x)` that consists of two distinct components: a slow part `g` that can be reused across multiple inputs and shared among various function evaluations, and a fast part `h` that is calculated for each `x` value.

```{code-cell} ipython3
def f(x):  # example function without caching
    """
    Integer part of `x` repeats and should be reused
    Decimal part requires a new computation
    """
    return g(int(x)) + h(x % 1)


def g(x):
    """Slow but reusable function"""
    from time import sleep

    sleep(random.randrange(5))
    return x**2


def h(x):
    """Fast function"""
    return x**3
```

### Using `adaptive.utils.daskify`

To simplify the process of using coroutines and caching with dask and Adaptive, we provide the {func}`adaptive.utils.daskify` decorator. This decorator can be used to parallelize functions with caching as well as functions without caching, making it a powerful tool for custom parallelization in Adaptive.

```{code-cell} ipython3
from dask.distributed import Client

import adaptive

client = await Client(asynchronous=True)


# The g function has caching enabled
g_dask = adaptive.utils.daskify(client, cache=True)(g)

# Can be used like a decorator too:
# >>> @adaptive.utils.daskify(client, cache=True)
# ... def g(x): ...

# The h function does not use caching
h_dask = adaptive.utils.daskify(client)(h)

# Now we need to rewrite `f(x)` to use `g` and `h` as coroutines


async def f_parallel(x):
    g_result = await g_dask(int(x))
    h_result = await h_dask(x % 1)
    return (g_result + h_result) ** 2


learner = adaptive.Learner1D(f_parallel, bounds=(-3.5, 3.5))
runner = adaptive.AsyncRunner(learner, loss_goal=0.01, ntasks=20)
runner.live_info()
```

Finally, we wait for the runner to finish, and then plot the result.

```{code-cell} ipython3
await runner.task
learner.plot()
```

### Step-by-step explanation of custom parallelization

Now let's dive into a detailed explanation of the process to understand how the {func}`adaptive.utils.daskify` decorator works.

In order to combine reuse of values of `g` with adaptive, we need to convert `f` into a dask graph by using `dask.delayed`.

```{code-cell} ipython3
from dask import delayed

# Convert g and h to dask.Delayed objects, such that they run in the Client
g, h = delayed(g), delayed(h)


@delayed
def f(x, y):
    return (x + y) ** 2
```

Next we define a computation using coroutines such that it reuses previously submitted tasks.

```{code-cell} ipython3
from dask.distributed import Client

client = await Client(asynchronous=True)

g_futures = {}


async def f_parallel(x):
    # Get or sumbit the slow function future
    if (g_future := g_futures.get(int(x))) is None:
        g_futures[int(x)] = g_future = client.compute(g(int(x)))

    future_f = client.compute(f(g_future, h(x % 1)))

    return await future_f
```

To run the adaptive evaluation we provide the asynchronous function to the `learner` and run it via `AsyncRunner` without specifying an executor.

```{code-cell} ipython3
learner = adaptive.Learner1D(f_parallel, bounds=(-3.5, 3.5))

runner = adaptive.AsyncRunner(learner, loss_goal=0.01, ntasks=20)
```

Finally we wait for the runner to finish, and then plot the result.

```{code-cell} ipython3
await runner.task
learner.plot()
```

## Using Runners from a script

Runners can also be used from a Python script independently of the notebook.

The simplest way to accomplish this is simply to use the {class}`~adaptive.BlockingRunner`:

```python
import adaptive


def f(x):
    return x


learner = adaptive.Learner1D(f, (-1, 1))

adaptive.BlockingRunner(learner, loss_goal=0.1)
```

If you use `asyncio` already in your script and want to integrate `adaptive` into it, then you can use the default {class}`~adaptive.Runner` as you would from a notebook.
If you want to wait for the runner to finish, then you can simply

```python
await runner.task
```

from within a coroutine.

[^download]: This notebook can be downloaded as **{nb-download}`tutorial.advanced-topics.ipynb`** and {download}`tutorial.advanced-topics.md`.
