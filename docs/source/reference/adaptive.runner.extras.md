# Runner extras

## Stopping Criteria

Runners allow you to specify the stopping criterion by providing
a `goal` as a function that takes the learner and returns a boolean: `False`
for "continue running" and `True` for "stop". This gives you a lot of flexibility
for defining your own stopping conditions, however we also provide some common
stopping conditions as a convenience. The `adaptive.runner.auto_goal` will
automatically create a goal based on simple input types, e.g., an int means
at least that many points are required and a float means that the loss has
to become lower or equal to that float.

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
