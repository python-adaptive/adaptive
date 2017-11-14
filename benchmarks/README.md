# adaptive benchmarks

Benchmarking adaptive with Airspeed Velocity.

## Usage

Airspeed Velocity manages building and Python conda environments by itself,
unless told otherwise. To run the benchmarks, you do not need to install a
development version of adaptive to your current Python environment.

Run ASV commands (record results and generate HTML):

```bash
cd benchmarks
asv run --skip-existing-commits --steps 10 ALL
asv publish
asv preview
```

More on how to use ``asv`` can be found in `ASV documentation`_
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.


## Writing benchmarks

See [`ASV documentation`](https://asv.readthedocs.io/) for basics on how to write benchmarks.

Some things to consider:

- The benchmark suite should be importable with any adaptive version.

- The benchmark parameters etc. should not depend on which adaptive version
  is installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's ``time_`` methods for benchmarking times rather than cooking up
  time measurements via ``time.clock``, even if it requires some juggling when
  writing the benchmark.

- Preparing arrays etc. should generally be put in the ``setup`` method rather
  than the ``time_`` methods, to avoid counting preparation time together with
  the time of the benchmarked operation.
