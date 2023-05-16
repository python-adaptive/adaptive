# ❓ FAQ: Frequently Asked Questions

## Where can I learn more about the algorithm used?

Read our [draft paper](https://gitlab.kwant-project.org/qt/adaptive-paper/builds/artifacts/master/file/paper.pdf?job=make) or the source code on [GitHub](https://github.com/python-adaptive/adaptive/).

## How do I get the data?

Check `learner.data`, `learner.to_numpy()`, or `learner.to_dataframe()`.

## How do I learn more than one value per point?

Use the {class}`adaptive.DataSaver`.

## My runner failed, how do I get the error message?

Check `runner.task.print_stack()`.

## How do I get a {class}`~adaptive.Learner2D`'s data on a grid?

Use `learner.interpolated_on_grid()` optionally with a argument `n` to specify the the amount of points in `x` and `y`.

## Why can I not use a `lambda` with a learner?

When using the {class}`~adaptive.Runner` the learner's function is evaluated in different Python processes.
Therefore, the `function` needs to be serialized (pickled) and send to the other Python processes; `lambda`s cannot be pickled.
Instead, you can probably use `functools.partial` to accomplish what you want to do.

## How do I run multiple runners?

Check out [Adaptive scheduler](http://adaptive-scheduler.readthedocs.io), which solves the following problem of needing to run more learners than you can run with a single runner.
It easily runs on tens of thousands of cores.

## What is the difference with FEM?

The main difference with FEM (Finite Element Method) is that one needs to globally update the mesh at every time step.

For Adaptive, we want to be able to parallelize the function evaluation and that requires an algorithm that can quickly return a new suggested point.
This means that, to minimize the time that Adaptive spends on adding newly calculated points to the data strucute, we only want to update the data of the points that are close to the new point.

## What is the difference with Bayesian optimization?

Adaptive and Bayesian optimization share some similarities, as both methods involve selecting new points for function evaluation based on the existing data. However, there are key differences between the two approaches.

1. **Goals**: Bayesian optimization is primarily focused on finding the minimum (or maximum) of a function, while Adaptive is more concerned with efficiently exploring and approximating the function across its domain.
2. **Global vs. Local**: Bayesian optimization typically relies on global fitting methods like Gaussian processes, which model the entire function using all available data points. This can provide a rigorous optimization strategy but may become computationally expensive for large datasets. In contrast, Adaptive uses local loss functions that depend only on the data local to a subdomain, which allows for more efficient parallelization and faster point suggestions.
3. **Computationally expensive data**: Bayesian optimization is well-suited for situations where the function evaluations are computationally expensive, whereas Adaptive is designed for low to intermediate cost functions. Regular grids work well for cheap data, and local adaptive algorithms like Adaptive are suitable for intermediate-cost scenarios.

In summary, Bayesian optimization is a good choice for more computationally demanding problems, while Adaptive is better suited for low to intermediate cost functions. Adaptive offers a more efficient and easily parallelizable approach due to its reliance on local loss functions.

# What is the difference with Kriging?

Kriging and Adaptive are both techniques used for learning and approximating functions, but they have different approaches and applications.

Kriging, also known as Gaussian process regression, is a geostatistical interpolation method used for predicting values at unsampled locations based on observed data points. It is based on fitting a Gaussian process model to the data, which provides a continuous function estimate along with uncertainty estimates. Kriging is particularly useful when dealing with spatial data and is commonly used in geostatistics, environmental sciences, and engineering.

Adaptive, on the other hand, is a Python library designed for parallel active learning of mathematical functions. It employs local loss functions to adaptively sample the function in the most interesting regions, concentrating more points in areas with high variation or rapid changes. Adaptive is suitable for low to intermediate cost functions and is designed for efficient parallelization and fast point suggestions.

In summary, the key differences between Kriging and Adaptive are:

1. **Methodology**: Kriging relies on global fitting using Gaussian processes, providing a continuous function estimate with uncertainty estimates. Adaptive uses local loss functions to adaptively sample the function in the most interesting regions.
2. **Applications**: Kriging is commonly used for spatial data interpolation in geostatistics and environmental sciences, while Adaptive is designed for parallel active learning of mathematical functions across various domains.
3. **Computation**: Kriging can be computationally expensive for large datasets due to its global fitting nature. Adaptive is more efficient and parallelizable, making it suitable for low to intermediate cost functions.

> For the stupid amongst us (myself included):
>
> - why would you to sample a function
> - what do you mean by sampling a function
> - what is adaptive sampling??
> - why?
> - how?
>
> What area of industry are you targeting with this (data industries or general programmers?)
> https://www.reddit.com/r/compsci/comments/133eqm2/comment/ji9r245/?utm_source=reddit&utm_medium=web2x&context=3


Great questions! I'm happy to clarify some concepts for you:

1.  **Why would you want to sample a function?** When working with functions, especially those that are computationally expensive or have complex behavior, it is often necessary to evaluate them at a finite set of points (sampling) to understand their properties or perform further analysis.

2.  **What does sampling a function mean?** Sampling a function means evaluating it at specific points in its domain. For example, if you have a function f(x) = x^2, sampling it might involve computing f(1), f(2), and f(3).

3.  **What is adaptive sampling?** Adaptive sampling is an approach where the choice of sampling points is not fixed but adapts based on the function's behavior. This allows us to concentrate more sampling points in regions of interest, such as areas with high variation or rapid changes, and fewer points where the function is relatively stable.

4.  **Why adaptive sampling?** The main advantage of adaptive sampling is efficiency. By focusing on the interesting regions of a function, we can reduce the number of required evaluations, potentially saving significant computation time and resources.

5.  **How does it work?** Our [Adaptive package](https://github.com/python-adaptive/adaptive/) intelligently selects the optimal points for sampling by analyzing existing data and adjusting on the fly. It can handle a variety of tasks, such as averaging stochastic functions, interpolating one and two-dimensional functions, and performing one-dimensional integration.

6.  **What area of industry are we targeting?** While Adaptive can be helpful for general programmers, it is particularly relevant for data-driven industries, scientific research, and engineering applications where understanding the behavior of complex functions is essential. Our package is designed to be flexible and user-friendly, making it applicable to a wide range of scenarios.

## What exactly is going on here?
> https://www.reddit.com/r/compsci/comments/133eqm2/comment/ji9oi5d/?utm_source=reddit&utm_medium=web2x&context=3

Adaptive is a Python package that helps you sample functions in a smart way. 🧠 Instead of evaluating a function on a fixed grid, Adaptive automatically samples the function more densely in regions where it is more "interesting" or changing rapidly. 📈

This approach has been super useful in my own work in quantum physics, where doing a lot of heavy calculations is common. By using Adaptive, I was able to speed up these calculations and improve the efficiency of my work. 💪

In the videos provided, Adaptive is shown in action on two different types of functions. The first video shows the adaptive sampling of a 2D function 🗺️, while the second video demonstrates the adaptive sampling of a 1D function 📏. In both cases, Adaptive automatically focuses on the most important parts of the function to save time and computational resources. 🕓💻

When comparing Adaptive's sampling method to uniform sampling, it's clear that Adaptive is way more efficient. ⚡ It concentrates computational resources on the regions of the function where more information is needed, leading to better results with fewer samples. 🎯 This is super valuable in situations where computational resources are limited or the calculations are time-consuming. 🔋⌛

## ELI5 version
> https://www.reddit.com/r/dataisbeautiful/comments/133edlr/comment/ji9oysk/?utm_source=reddit&utm_medium=web2x&context=3

Imagine you have a drawing with lots of hills and valleys, and you want to understand the shape of the landscape. Instead of measuring the height at every single point, Adaptive helps you measure the height at the most important points. It focuses on areas where the hills and valleys change a lot, so you can understand the drawing with fewer measurements.

This is useful because it saves time and resources, especially when measuring the height is difficult or takes a long time. Adaptive can be used by researchers, programmers, and others who need to understand how functions or data change in different situations.

## How does this work for more output dimensions?
> https://www.reddit.com/r/dataisbeautiful/comments/133edlr/comment/jia2bbv/?utm_source=reddit&utm_medium=web2x&context=3

Your approach of using Gaussian process regression and sampling based on uncertainty is quite similar to Bayesian optimization, which also has some similarities with Adaptive. However, there are some key differences.
Adaptive is more focused on exploring the function rather than minimizing it. It relies on local properties of the data rather than fitting it globally. This makes it computationally less expensive than Gaussian processes, which can become challenging to work with when dealing with tens of thousands of points.
As for output dimensions, Adaptive is primarily designed for low-dimensional functions. It works with ND (see https://adaptive.readthedocs.io/en/latest/tutorial/tutorial.LearnerND.html), however, due to the curse of dimensionality, it won't work well in high dimensional spaces.

Bayesian optimization might be more suitable when dealing with higher output dimensions, but it comes with the cost of increased computational complexity.
In summary, Bayesian optimization (using Gaussian process regression) is a good choice for computationally expensive data, regular grids for really cheap data, and local adaptive algorithms like Adaptive are somewhere in the middle.

## What is the difference with adaptive meshing in CFD or computer graphics?

Adaptive meshing in Computational Fluid Dynamics (CFD) or computer graphics and the Adaptive library both involve refining the representation of a function or a system based on its behavior, but they serve different purposes and use different techniques.

Adaptive meshing in CFD or computer graphics refers to the process of refining the mesh or grid used to approximate and simulate spatial phenomena, such as fluid flow or 3D surfaces. The mesh is adapted to concentrate more elements in regions with high gradients, rapid changes, or complex geometries, resulting in a more accurate representation of the system with fewer computational resources. This adaptation often requires global updates, which can be computationally expensive. Adaptive meshing techniques are widely used in hydrodynamics, astrophysics, and computer graphics for rendering and modeling purposes.

The Adaptive library, on the other hand, is designed for parallel active learning of mathematical functions. It focuses on adaptively sampling functions in their interesting regions, rather than evaluating them on a dense grid. Adaptive uses local loss functions to determine the most interesting points to sample next, allowing for more efficient parallelization and fast point suggestions. It is primarily intended for low to intermediate cost functions and works best with low-dimensional spaces.

In summary, the key differences between adaptive meshing in CFD or computer graphics and the Adaptive library are:

1. **Purpose**: Adaptive meshing is used for refining the mesh or grid in spatial simulations and rendering, while the Adaptive library is designed for parallel active learning of mathematical functions.
2. **Technique**: Adaptive meshing involves global updates to the mesh, which can be computationally expensive. The Adaptive library uses local loss functions for efficient parallelization and fast point suggestions.
3. **Application domains**: Adaptive meshing is commonly used in hydrodynamics, astrophysics, and computer graphics, while the Adaptive library is suitable for various domains involving learning and approximation of mathematical functions.

## Can I use this to tune my hyper parameters for machine learning models?

Yes, you can use the Adaptive library to tune hyperparameters for machine learning models. However, it is important to note that Adaptive is best suited for low to intermediate cost functions and works best with low-dimensional spaces.

To use Adaptive for hyperparameter tuning, you would need to define a function that takes hyperparameters as input and returns a performance metric (e.g., validation accuracy, cross-entropy loss, etc.) as output. You can then use one of the Adaptive learners (e.g., `Learner1D`, `LearnerND`) to sample the hyperparameter space adaptively, focusing on the most interesting regions where the model's performance is improving.

It is worth mentioning that there are specialized tools and libraries designed explicitly for hyperparameter optimization in machine learning, such as Bayesian optimization libraries like Scikit-Optimize or Hyperopt. These libraries are tailored for high-dimensional spaces and can handle more complex optimization scenarios.

In summary, while it is possible to use Adaptive for hyperparameter tuning, it may not be the most efficient or optimal choice for high-dimensional and complex optimization problems. Specialized hyperparameter optimization libraries might offer better performance in such cases.

% I get "``concurrent.futures.process.BrokenProcessPool``: A process in the process pool was terminated abruptly while the future was running or pending." what does it mean?
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing a question that you think belongs here? Let us [know](https://github.com/python-adaptive/adaptive/issues/new).
