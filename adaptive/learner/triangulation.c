#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static int
get_tuple_elems(PyObject * tuple, double * first, double * second) {
    if (PyTuple_Size(tuple) < 2) {
        PyErr_SetString(PyExc_ValueError, "Tuple must be len 2 or more.");
        return 0;
    }

    PyObject * elem = PyTuple_GetItem(tuple, 0);
    *first = PyFloat_AsDouble(elem);
    if (first == NULL)
        return 0;
    elem = PyTuple_GetItem(tuple, 1);
    *second = PyFloat_AsDouble(elem);
    if (second == NULL)
        return 0;
    return 1;
}

static int
get_triple_tuple_elems(PyObject * tuple, double * first, double * second, double* third) {
    if (PyTuple_Size(tuple) < 3) {
        PyErr_SetString(PyExc_ValueError, "Tuple must be len 3 or more.");
        return 0;
    }

    PyObject * elem = PyTuple_GetItem(tuple, 0);
    *first = PyFloat_AsDouble(elem);
    if (first == NULL)
        return 0;
    elem = PyTuple_GetItem(tuple, 1);
    *second = PyFloat_AsDouble(elem);
    if (second == NULL)
        return 0;
    elem = PyTuple_GetItem(tuple, 2);
    *third = PyFloat_AsDouble(elem);
    if (third == NULL)
        return 0;
    return 1;
}

static PyObject *
fast_norm(PyObject *self, PyObject *args)
{
    PyObject* vect;
	if (!PyArg_ParseTuple(args, "O", &vect))
		return NULL;

    if (PyList_Check(vect)) {
        Py_ssize_t numElements = PyObject_Length(vect);
        if (numElements <= 0) {
            PyErr_SetString(PyExc_ValueError, "fast_norm requires a list of length 1 or greater.");
            return NULL;
        }
        double sum = 0;
        for (Py_ssize_t i = 0; i < numElements; ++i) {
            double val = PyFloat_AsDouble(PyList_GetItem(vect, i));
            if (PyErr_Occurred()) {
                return NULL;
            }
            sum += val * val;
        }
        return Py_BuildValue("f", sqrt(sum));
    }

    if (PyTuple_Check(vect)) {
        Py_ssize_t numElements = PyTuple_Size(vect);
        if (numElements <= 0) {
            PyErr_SetString(PyExc_ValueError, "fast_norm requires a tuple of length 1 or greater.");
            return NULL;
        }
        double sum = 0;
        for (Py_ssize_t i = 0; i < numElements; ++i) {
            double val = PyFloat_AsDouble(PyTuple_GetItem(vect, i));
            if (PyErr_Occurred()) {
                return NULL;
            }
            sum += val * val;
        }
        return Py_BuildValue("f", sqrt(sum));
    }

    if (PyArray_Check(vect)) {
        PyObject *npArray = PyArray_FROM_OTF(vect, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (npArray == NULL)
            return NULL;
        int nd = PyArray_NDIM(npArray);
        if (nd != 1)
            return NULL;
        npy_intp * shape = PyArray_DIMS(npArray);
        Py_ssize_t numElements = PyArray_SIZE(npArray);
        double* data = (double *) PyArray_DATA(npArray);
        double sum = 0;
        for (Py_ssize_t i = 0; i < numElements; ++i) {
            double val = data[i];
            sum += val * val;
        }
        return Py_BuildValue("f", sqrt(sum));
    }

}

static PyObject *
fast_2d_point_in_simplex(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * point, * simplex;
    double eps = 0.00000001;

	static char *kwlist[] = {"point", "simplex", "eps", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|d", kwlist, &point, &simplex, &eps))
		return NULL;

    if (!PyTuple_Check(point)) {
        PyErr_SetString(PyExc_ValueError, "fast_2d_point_in_simplex requires a tuple as its first argument.");
        return NULL;
    }

    if (!PyList_Check(simplex)) {
        PyErr_SetString(PyExc_ValueError, "fast_2d_point_in_simplex requires a python list as its argument.");
        return NULL;
    }

    if (PyObject_Length(simplex) < 3) {
        PyErr_SetString(PyExc_ValueError, "fast_2d_point_in_simplex requires simplex to contain at least 3 elements.");
        return NULL;
    }

    double px, py, p0x, p0y, p1x, p1y, p2x, p2y;
    if (!get_tuple_elems(point, &px, &py))
        return NULL;


    point = PyList_GetItem(simplex, 0);
    if (!get_tuple_elems(point, &p0x, &p0y))
        return NULL;

    point = PyList_GetItem(simplex, 1);
    if (!get_tuple_elems(point, &p1x, &p1y))
        return NULL;

    point = PyList_GetItem(simplex, 2);
    if (!get_tuple_elems(point, &p2x, &p2y))
        return NULL;

    double area = 0.5 * (-p1y * p2x + p0y * (p2x - p1x) + p1x * p2y + p0x * (p1y - p2y));
    double s = 1 / (2 * area) * (p0y * p2x + (p2y - p0y) * px - p0x * p2y + (p0x - p2x) * py);
    if ((s < -eps) || (s > 1 + eps))
        return Py_BuildValue("O", Py_False);
    double t = 1 / (2 * area) * (p0x * p1y + (p0y - p1y) * px - p0y * p1x + (p1x - p0x) * py);
    return Py_BuildValue("O", (t >= -eps) && (s + t <= 1 + eps) ? Py_True : Py_False);
}

static PyObject *
fast_2d_circumcircle(PyObject *self, PyObject *args)
{
    PyObject* points;
	if (!PyArg_ParseTuple(args, "O", &points))
		return NULL;

    double x0, y0, x1, y1, x2, y2;

    if (PyList_Check(points)) {
        Py_ssize_t numElements = PyObject_Length(points);
        if (numElements < 3) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a list of length 3 or greater.");
            return NULL;
        }
        if (!get_tuple_elems(PyList_GetItem(points, 0), &x0, &y0))
            return NULL;
        if (!get_tuple_elems(PyList_GetItem(points, 1), &x1, &y1))
            return NULL;
        if (!get_tuple_elems(PyList_GetItem(points, 2), &x2, &y2))
            return NULL;
    } else if (PyTuple_Check(points)) {
        Py_ssize_t numElements = PyTuple_Size(points);
        if (numElements < 3) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a tuple of length 3 or greater.");
            return NULL;
        }

        if (!get_tuple_elems(PyTuple_GetItem(points, 0), &x0, &y0))
            return NULL;
        if (!get_tuple_elems(PyTuple_GetItem(points, 1), &x1, &y1))
            return NULL;
        if (!get_tuple_elems(PyTuple_GetItem(points, 2), &x2, &y2))
            return NULL;
    } else if (PyArray_Check(points)) {
        int dims = PyArray_NDIM(points);
        if (dims != 2) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a two dimensional numpy array.");
            return NULL;
        }
        npy_intp * shape = PyArray_DIMS(points);
        if ((shape[0] < 3) && (shape[1] != 2)) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a numpy array of width 3, height 2.");
            return NULL;
        }
        double* values = (double *) PyArray_DATA(points);
        x0 = values[0];
        y0 = values[1];
        x1 = values[2];
        y1 = values[3];
        x2 = values[4];
        y2 = values[5];
    } else {
        PyErr_SetString(PyExc_ValueError, "Points must be a list, tuple, or numpy array.");
        return NULL;
    }

    x1 -= x0;
    y1 -= y0;

    x2 -= x0;
    y2 -= y0;

    double l1 = x1 * x1 + y1 * y1;
    double l2 = x2 * x2 + y2 * y2;
    double dx = l1 * y2 - l2 * y1;
    double dy = -l1 * x2 + l2 * x1;

    double aa = 2 * (x1 * y2 - x2 * y1);
    double x = dx / aa;
    double y = dy / aa;
    return Py_BuildValue("(ff)f", x + x0, y + y0, sqrt(x * x + y * y));
}

static PyObject *
fast_3d_circumcircle(PyObject *self, PyObject *args)
{
    PyObject* points;
	if (!PyArg_ParseTuple(args, "O", &points))
		return NULL;

    double x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3;

    if (PyList_Check(points)) {
        Py_ssize_t numElements = PyObject_Length(points);
        if (numElements < 4) {
            PyErr_SetString(PyExc_ValueError, "fast_3d_circumcircle requires a list of length 4 or greater.");
            return NULL;
        }
        if (!get_triple_tuple_elems(PyList_GetItem(points, 0), &x0, &y0, &z0))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 1), &x1, &y1, &z1))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 2), &x2, &y2, &z2))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 3), &x3, &y3, &z3))
            return NULL;
    } else if (PyTuple_Check(points)) {
        Py_ssize_t numElements = PyTuple_Size(points);
        if (numElements < 4) {
            PyErr_SetString(PyExc_ValueError, "fast_3d_circumcircle requires a tuple of length 4 or greater.");
            return NULL;
        }
        if (!get_triple_tuple_elems(PyTuple_GetItem(points, 0), &x0, &y0, &z0))
            return NULL;
        if (!get_triple_tuple_elems(PyTuple_GetItem(points, 1), &x1, &y1, &z1))
            return NULL;
        if (!get_triple_tuple_elems(PyTuple_GetItem(points, 2), &x2, &y2, &z2))
            return NULL;
        if (!get_triple_tuple_elems(PyTuple_GetItem(points, 3), &x3, &y3, &z3))
            return NULL;
    } else if (PyArray_Check(points)) {
        int dims = PyArray_NDIM(points);

        if (dims != 2) {
            PyErr_SetString(PyExc_ValueError, "fast_3d_circumcircle requires a two dimensional numpy array.");
            return NULL;
        }

        npy_intp * shape = PyArray_DIMS(points);

        if ((shape[0] < 4) && (shape[1] != 3)) {
            PyErr_SetString(PyExc_ValueError, "fast_3d_circumcircle requires a numpy array of width 4, height 3.");
            return NULL;
        }

        double* values = (double *) PyArray_DATA(points);
        x0 = values[0];
        y0 = values[1];
        z0 = values[2];
        x1 = values[3];
        y1 = values[4];
        z1 = values[5];
        x2 = values[6];
        y2 = values[7];
        z2 = values[8];
        x3 = values[9];
        y3 = values[10];
        z3 = values[11];
    } else {
        PyErr_SetString(PyExc_ValueError, "Points must be a list, tuple, or numpy array.");
        return NULL;
    }

    x1 -= x0;
    y1 -= y0;
    z1 -= z0;

    x2 -= x0;
    y2 -= y0;
    z2 -= z0;

    x3 -= x0;
    y3 -= y0;
    z3 -= z0;

    double l1 = x1 * x1 + y1 * y1 + z1 * z1;
    double l2 = x2 * x2 + y2 * y2 + z2 * z2;
    double l3 = x3 * x3 + y3 * y3 + z3 * z3;
    double dx = l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2);
    double dy = l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2);
    double dz = l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2);

    double aa = 2 * (x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2));
    double x = dx / aa;
    double y = -dy / aa;
    double z = dz / aa;

    return Py_BuildValue("(fff)f", x + x0, y + y0, z + z0, sqrt(x * x + y * y + z * z));
}


static PyMethodDef triangulation_functions[] = {
    {"fast_norm", fast_norm, METH_VARARGS,
		"Returns the norm of the given array. Requires one dimensional tuple, list, or numpy array.\n"
		"For large matrices, it is better to use numpy's linear algebra functions.\nAdditionally, if the values in the "
		"list are particularly small, squaring them may cause them to become zero.\nIf they're very large, squaring them "
		"may result in numerical overflow (in the case where they're above 10^150).\nCan not handle complex numbers."},
    {"fast_2d_circumcircle", fast_2d_circumcircle, METH_VARARGS,
		"Returns center and radius of the circle touching the first three points in the list.\nRequires a list, tuple,"
		 "or numpy array that is 3x2 in shape."},
    {"fast_3d_circumcircle", fast_3d_circumcircle, METH_VARARGS,
		"Returns center and radius of the sphere touching the first four points in the list.\nRequires a list, tuple,"
		 "or numpy array that is 4x3 in shape."},
    {"fast_2d_point_in_simplex", (PyCFunction) fast_2d_point_in_simplex, METH_VARARGS | METH_KEYWORDS,
		"Returns true if the given 2d point is in the simplex, minus some error eps."},
	{NULL}
};

PyDoc_STRVAR(module_doc, "Fast implementations of triangulation functions.");

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"triangulation",
	module_doc,
	-1,                 /* m_size */
	triangulation_functions,   /* m_methods */
	NULL,               /* m_reload (unused) */
	NULL,               /* m_traverse */
	NULL,               /* m_clear */
	NULL                /* m_free */
};

PyObject *
PyInit_triangulation(void)
{
    if(PyArray_API == NULL)
        import_array();
    return PyModule_Create(&moduledef);
}
