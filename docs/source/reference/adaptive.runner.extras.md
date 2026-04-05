# Runner extras

## Stopping Criteria

Runners allow you to specify the stopping criterion by providing
a `goal` as a function that takes the learner and returns a boolean: `False`
for "continue running" and `True` for "stop". This gives you a lot of flexibility
for defining your own stopping conditions, however we also provide some common
stopping conditions as a convenience. For example, to continue until the loss is below a threshold `x`,
you may specify `loss_goal=x`. Similarly, to continue until `n` points have been sampled, you may
specify `npoints_goal=n`. See the Runner docstring for details.

```{eval-rst}
.. autofunction:: adaptive.runner.auto_goal
```

```{eval-rst}
.. autofunction:: adaptive.runner.stop_after
```

## Simple executor

```{eval-rst}
.. autofunction:: adaptive.runner.simple
```

## Sequential excecutor

```{eval-rst}
.. autoclass:: adaptive.runner.SequentialExecutor

```

## Replay log

```{eval-rst}
.. autofunction:: adaptive.runner.replay_log
```
