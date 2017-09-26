import numpy as np
from cquad import Learner
from algorithm_4 import algorithm_4

def same_ivals(f, a, b, tol, verbose):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)
        l = Learner(f, bounds=(a, b), tol=tol)

        for i in range(nr_points):
            points, loss_improvement = l.choose_points(1)
            l.add_data(points, map(l.function, points))

        return l.equal(ivals, verbose=verbose)


def same_ivals_up_to(f, a, b, tol):
        igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)
        l = Learner(f, bounds=(a, b), tol=tol)
        j = 0
        equal_till = 0
        for i in range(nr_points):
            points, loss_improvement = l.choose_points(1)
            l.add_data(points, map(l.function, points))
            if not l._stack:
                try:
                    j += 1
                    if l.equal(ivals):
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
                             [f24, 0, 3, 1e-3],  # Not the same
                             [f63, 0, 1, 1e-10],  # Error
                             [fdiv, 0, 1, 1e-6]]):  # diverging error not implemented correctly
        print('\nFunction {}'.format(i))
        if same_ivals(*args, verbose=True):
            print(True)
        else:
            print(same_ivals_up_to(*args))

    np.seterr(**old_settings)
