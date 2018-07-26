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


def fast_volume_2d(points):
    cdef double x0, y0, x1, y1, x2, y2
    (x0, y0), (x1, y1), (x2, y2) = points
    x1 -= x0
    x2 -= x0
    y1 -= y0
    y2 -= y0

    return abs(x1*y2-y1*x2) * 0.5


def fast_volume_3d(points):
    cdef double x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = points
    x1 -= x0
    x2 -= x0
    x3 -= x0
    y1 -= y0
    y2 -= y0
    y3 -= y0
    z1 -= z0
    z2 -= z0
    z3 -= z0
    return abs(x1*(y2*z3-z2*y3)-x2*(y1*z3-z1*y3)+x3*(y1*z2-z1*y2)) * 0.16666666666



def fast_3d_circumcircle(points):
    """Compute the center and radius of the circumscribed shpere of a simplex.

    Parameters
    ----------
    points: 2D array-like
        the points of the triangle to investigate

    Returns
    -------
    tuple
        (center point : tuple(int), radius: int)
    """
    cdef double x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, l1, l2, l3, dx, dy, dz, aa, a, radius
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = points
    x1 -= x0
    x2 -= x0
    x3 -= x0
    y1 -= y0
    y2 -= y0
    y3 -= y0
    z1 -= z0
    z2 -= z0
    z3 -= z0

    l1 = x1 * x1 + y1 * y1 + z1 * z1
    l2 = x2 * x2 + y2 * y2 + z2 * z2
    l3 = x3 * x3 + y3 * y3 + z3 * z3

    # Compute some determinants:
    dx = (+ l1 * (y2 * z3 - z2 * y3)
          - l2 * (y1 * z3 - z1 * y3)
          + l3 * (y1 * z2 - z1 * y2))
    dy = (+ l1 * (x2 * z3 - z2 * x3)
          - l2 * (x1 * z3 - z1 * x3)
          + l3 * (x1 * z2 - z1 * x2))
    dz = (+ l1 * (x2 * y3 - y2 * x3)
          - l2 * (x1 * y3 - y1 * x3)
          + l3 * (x1 * y2 - y1 * x2))
    aa = (+ x1 * (y2 * z3 - z2 * y3)
          - x2 * (y1 * z3 - z1 * y3)
          + x3 * (y1 * z2 - z1 * y2))
    a = 2 * aa

    x =  dx / a
    y = -dy / a
    z =  dz / a
    radius = fast_norm((x,y,z))

    return (x + x0, y + y0, z + z0), radius
