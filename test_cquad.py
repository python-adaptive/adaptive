import operator

import numpy as np

from cquad import Learner
from algorithm_4 import algorithm_4

def __eq__(self, other, *, verbose=True):
    variables = []
    for slot in self.__slots__:
        try:
            eq = np.allclose(getattr(self, slot), getattr(other, slot), equal_nan=True)
        except:
            eq = getattr(self, slot) == getattr(other, slot)
        if not eq and verbose:
            print(slot, getattr(self, slot) - getattr(other, slot))
        variables.append(eq)
    return all(variables)


def same_ivals(old, new, *, verbose=False):
    old = sorted(old, key=operator.attrgetter('a'))
    new = sorted(new, key=operator.attrgetter('a'))
    try:
        return [__eq__(ival1, ival2, verbose=verbose) for ival1, ival2 in zip(old, new)]
    except:
        return [False]


def compare_algo(f, a, b, tol, verbose=False):
    verbose = False
    l = Learner(f, bounds=(a, b), tol=tol)
    igral, err, nr_points, ivals = algorithm_4(f, a, b, tol)
    j = 0
    for i in range(nr_points):
        points, loss_improvement = l.choose_points(1)
        l.add_data(points, map(l.function, points))
        if not l._stack:
            *_, ivals = algorithm_4(f, a, b, tol, until_iteration=j)
            all_the_same = all(same_ivals(ivals, l.ivals))
            if all_the_same:
                if verbose:
                    print('Identical till point number: {}, which are {} full cycles in the while loop.'.format(i + 1, j + 1))
                if i + 1 == nr_points:
                    return True
                j += 1

if __name__ == '__main__':
    old_settings = np.seterr(all='ignore')

    from algorithm_4 import f0, f7, f24, f21, f63, fdiv
    for i, args in enumerate([[f0, 0, 3, 1e-5],
                              [f7, 0, 1, 1e-6],
                              [f24, 0, 3, 1e-3], # Not the same, no error
                              [f21, 0, 1, 1e-3],
                              [f63, 0, 1, 1e-10], # Error
                              [fdiv, 0, 1, 1e-6]]):
        print(compare_algo(*args))

    np.seterr(**old_settings)