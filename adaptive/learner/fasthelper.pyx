import math


def fast_norm(v):
    # notice this method can be even more optimised
    if len(v) == 2:
        return math.sqrt(v[0] * v[0] + v[1] * v[1])
    if len(v) == 3:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    sum = 0
    for i in v:
        sum += i*i
    return math.sqrt(sum)



def fast_2d_circumcircle(points):
    """Compute the center and radius of the circumscribed circle of a triangle

    Parameters
    ----------
    points: 2D array-like
        the points of the triangle to investigate

    Returns
    -------
    tuple
        (center point : tuple(int), radius: int)
    """

    # transform to relative coordinates
    cdef double x0,y0,x1,y1,x2,y2, l1,l2,dx,dy,aa,a,x,y,radius

    (x0, y0), (x1, y1), (x2, y2) = points
    x1 -= x0
    y1 -= y0
    x2 -= x0
    y2 -= y0

    # compute the length squared
    l1 = x1 * x1 + y1 * y1
    l2 = x2 * x2 + y2 * y2

    # compute some determinants
    dx = + l1 * y2 - l2 * y1
    dy = - l1 * x2 + l2 * x1
    a = (+ x1 * y2 - x2 * y1) * 2

    # compute center
    x = dx / a
    y = dy / a
    radius = math.sqrt(x*x + y*y)  # radius = norm([x, y])

    return (x + x0, y + y0), radius
