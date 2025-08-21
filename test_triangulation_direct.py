#!/usr/bin/env python3
"""Direct comparison of Python vs Rust triangulation performance."""

import time

import numpy as np
from adaptive_rust import RustTriangulation

from adaptive.learner.triangulation import Triangulation as PyTriangulation


def benchmark_triangulation_creation(n_points=100, dim=2):
    """Benchmark triangulation creation and point addition."""
    print(f"\nBenchmarking with {n_points} points in {dim}D:")
    print("-" * 50)

    # Generate random points
    np.random.seed(42)
    initial_points = np.random.rand(dim + 1, dim)  # Initial simplex
    additional_points = np.random.rand(n_points, dim)

    # Python implementation
    print("Python Triangulation:")
    start = time.perf_counter()
    py_tri = PyTriangulation(initial_points)
    py_create_time = time.perf_counter() - start
    print(f"  Creation time: {py_create_time * 1000:.3f}ms")

    start = time.perf_counter()
    for _i, pt in enumerate(additional_points):
        try:
            py_tri.add_point(pt)
        except ValueError:
            pass  # Point already exists
    py_add_time = time.perf_counter() - start
    print(f"  Adding {n_points} points: {py_add_time * 1000:.3f}ms")
    print(f"  Final simplices: {len(py_tri.simplices)}")
    py_total = py_create_time + py_add_time

    # Rust implementation
    print("\nRust Triangulation:")
    start = time.perf_counter()
    rust_tri = RustTriangulation(initial_points)
    rust_create_time = time.perf_counter() - start
    print(f"  Creation time: {rust_create_time * 1000:.3f}ms")

    start = time.perf_counter()
    for _i, pt in enumerate(additional_points):
        try:
            rust_tri.add_point(pt.tolist())
        except ValueError:
            pass  # Point already exists
    rust_add_time = time.perf_counter() - start
    print(f"  Adding {n_points} points: {rust_add_time * 1000:.3f}ms")
    print(f"  Final simplices: {len(rust_tri.get_simplices())}")
    rust_total = rust_create_time + rust_add_time

    # Speedup
    speedup = py_total / rust_total
    print(f"\nSpeedup: {speedup:.2f}x")

    return py_total, rust_total, speedup


def benchmark_point_location(n_points=100, n_queries=1000):
    """Benchmark point location performance."""
    print(f"\nBenchmarking point location with {n_points} points, {n_queries} queries:")
    print("-" * 50)

    # Create triangulations with points
    np.random.seed(42)
    dim = 2
    initial_points = np.random.rand(dim + 1, dim)
    additional_points = np.random.rand(n_points, dim)

    # Build Python triangulation
    py_tri = PyTriangulation(initial_points)
    for pt in additional_points:
        try:
            py_tri.add_point(pt)
        except ValueError:
            pass

    # Build Rust triangulation
    rust_tri = RustTriangulation(initial_points)
    for pt in additional_points:
        try:
            rust_tri.add_point(pt.tolist())
        except ValueError:
            pass

    # Generate query points
    query_points = np.random.rand(n_queries, dim)

    # Python point location
    print("Python point location:")
    start = time.perf_counter()
    py_found = 0
    for pt in query_points:
        simplex = py_tri.locate_point(pt)
        if simplex:
            py_found += 1
    py_time = time.perf_counter() - start
    print(f"  Time: {py_time * 1000:.3f}ms")
    print(f"  Points found: {py_found}/{n_queries}")

    # Rust point location
    print("\nRust point location:")
    start = time.perf_counter()
    rust_found = 0
    for pt in query_points:
        simplex = rust_tri.locate_point(pt.tolist())
        if simplex:
            rust_found += 1
    rust_time = time.perf_counter() - start
    print(f"  Time: {rust_time * 1000:.3f}ms")
    print(f"  Points found: {rust_found}/{n_queries}")

    speedup = py_time / rust_time
    print(f"\nSpeedup: {speedup:.2f}x")

    return py_time, rust_time, speedup


def benchmark_scaling():
    """Benchmark how performance scales with number of points."""
    print("\n" + "=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)

    point_counts = [50, 100, 200, 500, 1000]
    creation_speedups = []
    location_speedups = []

    for n in point_counts:
        _, _, speedup_create = benchmark_triangulation_creation(n, dim=2)
        creation_speedups.append(speedup_create)

        _, _, speedup_locate = benchmark_point_location(n, n_queries=100)
        location_speedups.append(speedup_locate)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nCreation/Addition Speedups:")
    for n, s in zip(point_counts, creation_speedups):
        print(f"  {n:4d} points: {s:.2f}x")

    print("\nPoint Location Speedups:")
    for n, s in zip(point_counts, location_speedups):
        print(f"  {n:4d} points: {s:.2f}x")

    avg_creation = sum(creation_speedups) / len(creation_speedups)
    avg_location = sum(location_speedups) / len(location_speedups)
    print("\nAverage speedups:")
    print(f"  Creation/Addition: {avg_creation:.2f}x")
    print(f"  Point Location: {avg_location:.2f}x")


def benchmark_high_dimensions():
    """Benchmark performance in higher dimensions."""
    print("\n" + "=" * 60)
    print("HIGH-DIMENSIONAL BENCHMARK")
    print("=" * 60)

    for dim in [2, 3, 4, 5]:
        n_points = 50  # Fewer points for higher dimensions
        print(f"\n{dim}D space with {n_points} points:")
        print("-" * 40)

        # Generate points
        np.random.seed(42)
        initial_points = np.random.rand(dim + 1, dim)
        additional_points = np.random.rand(n_points, dim)

        # Python
        start = time.perf_counter()
        py_tri = PyTriangulation(initial_points)
        for pt in additional_points:
            try:
                py_tri.add_point(pt)
            except ValueError:
                pass
        py_time = time.perf_counter() - start
        print(f"  Python: {py_time * 1000:.3f}ms")

        # Rust
        start = time.perf_counter()
        rust_tri = RustTriangulation(initial_points)
        for pt in additional_points:
            try:
                rust_tri.add_point(pt.tolist())
            except ValueError:
                pass
        rust_time = time.perf_counter() - start
        print(f"  Rust: {rust_time * 1000:.3f}ms")

        speedup = py_time / rust_time
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("DIRECT TRIANGULATION PERFORMANCE COMPARISON")
    print("=" * 60)

    # Basic benchmarks
    benchmark_triangulation_creation(100, dim=2)
    benchmark_point_location(100, 1000)

    # Scaling benchmark
    benchmark_scaling()

    # High-dimensional benchmark
    benchmark_high_dimensions()

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETED!")
    print("=" * 60)
