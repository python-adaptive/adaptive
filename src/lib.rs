use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use nalgebra::DMatrix;

mod geometry;
mod triangulation;

// Import geometry functions (used internally, not directly exposed)

/// Fast determinant calculation for 2x2 and 3x3 matrices, falls back to general for larger
#[pyfunction]
#[pyo3(name = "fast_det")]
fn py_fast_det(_py: Python, matrix: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err("Matrix must be square"));
    }

    let result = if shape[0] == 2 && shape[1] == 2 {
        let m = matrix.as_array();
        m[[0, 0]] * m[[1, 1]] - m[[1, 0]] * m[[0, 1]]
    } else if shape[0] == 3 && shape[1] == 3 {
        let m = matrix.as_array();
        let a = m[[0, 0]];
        let b = m[[0, 1]];
        let c = m[[0, 2]];
        let d = m[[1, 0]];
        let e = m[[1, 1]];
        let f = m[[1, 2]];
        let g = m[[2, 0]];
        let h = m[[2, 1]];
        let i = m[[2, 2]];
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    } else {
        // For larger matrices, convert to nalgebra and compute
        let n = shape[0];
        let mut m = DMatrix::<f64>::zeros(n, n);
        let array = matrix.as_array();
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = array[[i, j]];
            }
        }
        m.determinant()
    };

    Ok(result)
}

/// Fast vector norm calculation optimized for 2D and 3D vectors
#[pyfunction]
#[pyo3(name = "fast_norm")]
fn py_fast_norm(v: Vec<f64>) -> f64 {
    geometry::fast_norm(&v)
}

/// Compute the center and radius of the circumsphere of a simplex
#[pyfunction]
#[pyo3(name = "circumsphere")]
fn py_circumsphere(py: Python, points: PyReadonlyArray2<f64>) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    let shape = points.shape();
    let ndim = shape[1];
    let npoints = shape[0];

    if npoints != ndim + 1 {
        return Err(PyValueError::new_err(
            format!("Expected {} points for {}-dimensional circumsphere, got {}",
                    ndim + 1, ndim, npoints)
        ));
    }

    // Convert to Vec<Vec<f64>>
    let mut pts = Vec::new();
    let array = points.as_array();
    for i in 0..npoints {
        let mut point = Vec::new();
        for j in 0..ndim {
            point.push(array[[i, j]]);
        }
        pts.push(point);
    }

    let (center, radius) = geometry::circumsphere(&pts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let center_array = PyArray1::from_vec_bound(py, center);
    Ok((center_array.into(), radius))
}

/// Check if a point is inside a simplex
#[pyfunction]
#[pyo3(name = "point_in_simplex", signature = (point, simplex, eps=None))]
fn py_point_in_simplex(
    point: Vec<f64>,
    simplex: Vec<Vec<f64>>,
    eps: Option<f64>
) -> PyResult<bool> {
    let eps = eps.unwrap_or(1e-8);

    // Validate dimensions
    if simplex.is_empty() {
        return Err(PyValueError::new_err("Simplex cannot be empty"));
    }

    let dim = point.len();
    if simplex.len() != dim + 1 {
        return Err(PyValueError::new_err(
            format!("Simplex must have {} points for {}-dimensional space", dim + 1, dim)
        ));
    }

    for pt in &simplex {
        if pt.len() != dim {
            return Err(PyValueError::new_err("All simplex points must have same dimension"));
        }
    }

    Ok(geometry::point_in_simplex(&point, &simplex, eps))
}

/// Fast 2D point in simplex test
#[pyfunction]
#[pyo3(name = "fast_2d_point_in_simplex", signature = (point, simplex, eps=None))]
fn py_fast_2d_point_in_simplex(
    point: (f64, f64),
    simplex: Vec<(f64, f64)>,
    eps: Option<f64>
) -> PyResult<bool> {
    let eps = eps.unwrap_or(1e-8);

    if simplex.len() != 3 {
        return Err(PyValueError::new_err("2D simplex must have exactly 3 points"));
    }

    Ok(geometry::fast_2d_point_in_simplex(point, &simplex, eps))
}

/// Calculate the volume of a simplex
#[pyfunction]
#[pyo3(name = "volume")]
fn py_volume(_py: Python, simplex: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let shape = simplex.shape();
    let n_points = shape[0];
    let dim = shape[1];

    if n_points != dim + 1 {
        return Err(PyValueError::new_err(
            format!("Expected {} points for {}-dimensional simplex, got {}",
                    dim + 1, dim, n_points)
        ));
    }

    // Convert to Vec<Vec<f64>>
    let mut points = Vec::new();
    let array = simplex.as_array();
    for i in 0..n_points {
        let mut point = Vec::new();
        for j in 0..dim {
            point.push(array[[i, j]]);
        }
        points.push(point);
    }

    Ok(geometry::volume(&points))
}

/// Calculate the volume of a simplex in a higher-dimensional embedding
#[pyfunction]
#[pyo3(name = "simplex_volume_in_embedding")]
fn py_simplex_volume_in_embedding(_py: Python, vertices: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let shape = vertices.shape();
    let n_vertices = shape[0];
    let dim = shape[1];

    // Convert to Vec<Vec<f64>>
    let mut verts = Vec::new();
    let array = vertices.as_array();
    for i in 0..n_vertices {
        let mut point = Vec::new();
        for j in 0..dim {
            point.push(array[[i, j]]);
        }
        verts.push(point);
    }

    geometry::simplex_volume_in_embedding(&verts)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Python module definition
#[pymodule]
fn adaptive_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_fast_det, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_norm, m)?)?;
    m.add_function(wrap_pyfunction!(py_circumsphere, m)?)?;
    m.add_function(wrap_pyfunction!(py_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_2d_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplex_volume_in_embedding, m)?)?;

    // Add the RustTriangulation class
    m.add_class::<triangulation::RustTriangulation>()?;

    Ok(())
}
