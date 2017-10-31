import numpy as np
from adaptive.learner import IntegratorLearner
from algorithm_4 import algorithm_4

def same_ivals(f, a, b, tol, verbose):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

        learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
        for i in range(nr_points):
            points, loss_improvement = learner.choose_points(1)
            learner.add_data(points, map(learner.function, points))
        if verbose:
            print('igral diff, ', learner.igral-igral, 'err diff', learner.err - err)
        return learner.equal(ivals, verbose=verbose)


def same_ivals_up_to(f, a, b, tol):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)

        learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
        j = 0
        equal_till = 0
        for i in range(nr_points):
            points, loss_improvement = learner.choose_points(1)
            learner.add_data(points, map(learner.function, points))
            if not learner._stack:
                try:
                    j += 1
                    if learner.equal(ivals):
                        equal_till = i + 1
                except:
                    all_equal = False

        return 'equal_till nr_points={} of {}'.format(equal_till, nr_points)

if __name__ == '__main__':
    old_settings = np.seterr(all='ignore')
    from algorithm_4 import f0, f7, f24, f21, f63, fdiv
    for i, args in enumerate([[f0, 0, 3, 1e-5],
                              [f7, 0, 1, 1e-6],
                              [f21, 0, 1, 1e-3],
                              [f24, 0, 3, 1e-3],
                              [f63, 0, 1, 1e-10]]):
        print('\nFunction {}'.format(i))
        if same_ivals(*args, verbose=True):
            print(True)
        else:
            print(same_ivals_up_to(*args))
    
    # This function should raise a DivergentIntegralError.
    print('Function ', i+1)
    f, a, b, tol = [fdiv, 0, 1, 1e-6]
    try:
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)
    except Exception:
        print('The integral is diverging.')

    try:
        learner = IntegratorLearner(f, bounds=(a, b), tol=tol)
        for i in range(nr_points):
            points, loss_improvement = learner.choose_points(1)
            learner.add_data(points, map(learner.function, points))
    except Exception:
        print('The integral is diverging.')

    np.seterr(**old_settings)
