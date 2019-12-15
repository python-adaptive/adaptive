#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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
            PyErr_SetString(PyExc_ValueError, "fast_norm requires a list of length 1 or greater.");
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

static int get_tuple_elems(PyObject * tuple, double * first, double * second) {
    PyObject * elem = PyTuple_GetItem(tuple, 0);
    if (elem == NULL)
        return 0;
    *first = PyFloat_AsDouble(elem);
    elem = PyTuple_GetItem(tuple, 1);
    if (elem == NULL)
        return 0;
    *second = PyFloat_AsDouble(elem);
    return 1;
}

static int get_triple_tuple_elems(PyObject * tuple, double * first, double * second, double* third) {
    PyObject * elem = PyTuple_GetItem(tuple, 0);
    if (elem == NULL)
        return 0;
    *first = PyFloat_AsDouble(elem);
    elem = PyTuple_GetItem(tuple, 1);
    if (elem == NULL)
        return 0;
    *second = PyFloat_AsDouble(elem);
    elem = PyTuple_GetItem(tuple, 2);
    if (elem == NULL)
        return 0;
    *third = PyFloat_AsDouble(elem);
    return 1;
}

static PyObject *
fast_2d_point_in_simplex(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject* point, * simplex;
    double eps = 0.00000001;

    // SystemError: bad format char passed to Py_BuildValue ??
	static char *kwlist[] = {"point", "simplex", "eps", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|d", kwlist, &point, &simplex, &eps))
		return NULL;


    if (!PyTuple_Check(point))
        return NULL;

    if (!PyList_Check(simplex))
        return NULL;

    double px, py, p0x, p0y, p1x, p1y, p2x, p2y;
    if (!get_tuple_elems(point, &px, &py))
        return NULL;

    point = PyList_GetItem(simplex, 0);
    if (point == NULL)
        return NULL;

    if (!get_tuple_elems(point, &p0x, &p0y))
        return NULL;

    point = PyList_GetItem(simplex, 1);
    if (point == NULL)
        return NULL;

    if (!get_tuple_elems(point, &p1x, &p1y))
        return NULL;

    point = PyList_GetItem(simplex, 2);
    if (point == NULL)
        return NULL;

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

    if (PyList_Check(points)) {
        Py_ssize_t numElements = PyObject_Length(points);
        if (numElements <= 0) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a list of length 1 or greater.");
            return NULL;
        }
        double x0, y0, x1, y1, x2, y2;
        if (!get_tuple_elems(PyList_GetItem(points, 0), &x0, &y0))
            return NULL;
        if (!get_tuple_elems(PyList_GetItem(points, 1), &x1, &y1))
            return NULL;
        if (!get_tuple_elems(PyList_GetItem(points, 2), &x2, &y2))
            return NULL;

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
    return NULL;
}

static PyObject *
fast_3d_circumcircle(PyObject *self, PyObject *args)
{
    PyObject* points;
	if (!PyArg_ParseTuple(args, "O", &points))
		return NULL;

    if (PyList_Check(points)) {
        Py_ssize_t numElements = PyObject_Length(points);
        if (numElements <= 0) {
            PyErr_SetString(PyExc_ValueError, "fast_2d_circumcircle requires a list of length 1 or greater.");
            return NULL;
        }
        double x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 0), &x0, &y0, &z0))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 1), &x1, &y1, &z1))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 2), &x2, &y2, &z2))
            return NULL;
        if (!get_triple_tuple_elems(PyList_GetItem(points, 3), &x3, &y3, &z3))
            return NULL;

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
        double dy = -l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2);
        double dz = l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2);

        double aa = 2 * (x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2));
        double x = dx / aa;
        double y = dy / aa;
        double z = dz / aa;
        return Py_BuildValue("(fff)f", x + x0, y + y0, z + z0, sqrt(x * x + y * y + z * z));
    }
    return NULL;
}


static PyMethodDef triangulation_functions[] = {
    {"fast_norm", fast_norm, METH_VARARGS,
		"Returns the norm of the given array."},
    {"fast_2d_circumcircle", fast_2d_circumcircle, METH_VARARGS,
		"Returns the norm of the given array."},
    {"fast_3d_circumcircle", fast_3d_circumcircle, METH_VARARGS,
		"Returns the norm of the given array."},
    {"fast_2d_point_in_simplex", (PyCFunction) fast_2d_point_in_simplex, METH_VARARGS | METH_KEYWORDS,
		"Refer to docs."},
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
