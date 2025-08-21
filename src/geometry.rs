use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GeometryError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Singular matrix")]
    SingularMatrix,
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Fast norm calculation optimized for 2D and 3D vectors
pub fn fast_norm(v: &[f64]) -> f64 {
    match v.len() {
        2 => (v[0] * v[0] + v[1] * v[1]).sqrt(),
        3 => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
        _ => {
            let sum: f64 = v.iter().map(|x| x * x).sum();
            sum.sqrt()
        }
    }
}

/// Fast determinant calculation for small matrices
pub fn fast_det(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 2 {
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    } else if n == 3 {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[0][2];
        let d = matrix[1][0];
        let e = matrix[1][1];
        let f = matrix[1][2];
        let g = matrix[2][0];
        let h = matrix[2][1];
        let i = matrix[2][2];
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    } else {
        // For larger matrices, use nalgebra
        let mut m = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = matrix[i][j];
            }
        }
        m.determinant()
    }
}

/// Fast 2D point in triangle test
pub fn fast_2d_point_in_simplex(point: (f64, f64), simplex: &[(f64, f64)], eps: f64) -> bool {
    let (p0x, p0y) = simplex[0];
    let (p1x, p1y) = simplex[1];
    let (p2x, p2y) = simplex[2];
    let (px, py) = point;

    let area = 0.5 * (-p1y * p2x + p0y * (p2x - p1x) + p1x * p2y + p0x * (p1y - p2y));

    let s = 1.0 / (2.0 * area) * (p0y * p2x + (p2y - p0y) * px - p0x * p2y + (p0x - p2x) * py);
    if s < -eps || s > 1.0 + eps {
        return false;
    }

    let t = 1.0 / (2.0 * area) * (p0x * p1y + (p0y - p1y) * px - p0y * p1x + (p1x - p0x) * py);

    t >= -eps && (s + t) <= 1.0 + eps
}

/// General point in simplex test
pub fn point_in_simplex(point: &[f64], simplex: &[Vec<f64>], eps: f64) -> bool {
    if point.len() == 2 && simplex.len() == 3 {
        // Use fast 2D version if possible
        let p = (point[0], point[1]);
        let s: Vec<(f64, f64)> = simplex.iter()
            .map(|v| (v[0], v[1]))
            .collect();
        return fast_2d_point_in_simplex(p, &s, eps);
    }

    let dim = point.len();
    let x0 = &simplex[0];

    // Build matrix of vectors from x0 to other vertices
    let mut matrix = DMatrix::<f64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            matrix[(j, i)] = simplex[i + 1][j] - x0[j];
        }
    }

    // Build RHS vector
    let mut b = DVector::<f64>::zeros(dim);
    for i in 0..dim {
        b[i] = point[i] - x0[i];
    }

    // Solve for barycentric coordinates
    match matrix.lu().solve(&b) {
        Some(alpha) => {
            // Check if all alphas are positive and sum < 1
            let sum: f64 = alpha.iter().sum();
            alpha.iter().all(|&a| a > -eps) && sum < 1.0 + eps
        }
        None => false, // Singular matrix means degenerate simplex
    }
}

/// Fast 2D circumcircle calculation
pub fn fast_2d_circumcircle(points: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let p0 = &points[0];
    let p1 = &points[1];
    let p2 = &points[2];

    // Transform to relative coordinates
    let x1 = p1[0] - p0[0];
    let y1 = p1[1] - p0[1];
    let x2 = p2[0] - p0[0];
    let y2 = p2[1] - p0[1];

    // Compute length squared
    let l1 = x1 * x1 + y1 * y1;
    let l2 = x2 * x2 + y2 * y2;

    // Compute determinants
    let dx = l1 * y2 - l2 * y1;
    let dy = -l1 * x2 + l2 * x1;
    let a = 2.0 * (x1 * y2 - x2 * y1);

    // Compute center
    let x = dx / a;
    let y = dy / a;
    let radius = (x * x + y * y).sqrt();

    let center = vec![x + p0[0], y + p0[1]];
    (center, radius)
}

/// Fast 3D circumsphere calculation
pub fn fast_3d_circumsphere(points: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let p0 = &points[0];
    let p1 = &points[1];
    let p2 = &points[2];
    let p3 = &points[3];

    // Transform to relative coordinates
    let x1 = p1[0] - p0[0];
    let y1 = p1[1] - p0[1];
    let z1 = p1[2] - p0[2];
    let x2 = p2[0] - p0[0];
    let y2 = p2[1] - p0[1];
    let z2 = p2[2] - p0[2];
    let x3 = p3[0] - p0[0];
    let y3 = p3[1] - p0[1];
    let z3 = p3[2] - p0[2];

    let l1 = x1 * x1 + y1 * y1 + z1 * z1;
    let l2 = x2 * x2 + y2 * y2 + z2 * z2;
    let l3 = x3 * x3 + y3 * y3 + z3 * z3;

    // Compute determinants
    let dx = l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2);
    let dy = l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2);
    let dz = l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2);
    let aa = x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2);
    let a = 2.0 * aa;

    let cx = dx / a;
    let cy = -dy / a;
    let cz = dz / a;
    let radius = (cx * cx + cy * cy + cz * cz).sqrt();

    let center = vec![cx + p0[0], cy + p0[1], cz + p0[2]];
    (center, radius)
}

/// General N-dimensional circumsphere calculation
pub fn circumsphere(points: &[Vec<f64>]) -> Result<(Vec<f64>, f64), GeometryError> {
    let dim = points.len() - 1;

    // Use optimized versions for 2D and 3D
    if dim == 2 {
        return Ok(fast_2d_circumcircle(points));
    }
    if dim == 3 {
        return Ok(fast_3d_circumsphere(points));
    }

    // General case using determinants
    let mut mat = vec![vec![0.0; dim + 2]; dim + 1];

    for (i, pt) in points.iter().enumerate() {
        let sum_sq: f64 = pt.iter().map(|x| x * x).sum();
        mat[i][0] = sum_sq;
        for j in 0..dim {
            mat[i][j + 1] = pt[j];
        }
        mat[i][dim + 1] = 1.0;
    }

    // Compute center coordinates using Cramer's rule
    let mut center = vec![0.0; dim];

    // First compute the denominator determinant (without first column)
    let mut denom_mat = DMatrix::<f64>::zeros(dim + 1, dim + 1);
    for i in 0..=dim {
        for j in 0..=dim {
            denom_mat[(i, j)] = mat[i][j + 1];
        }
    }
    let denom = denom_mat.determinant();

    if denom.abs() < 1e-15 {
        return Err(GeometryError::SingularMatrix);
    }

    let a = 1.0 / (2.0 * denom);
    let mut factor = a;

    for coord_idx in 0..dim {
        // Build matrix with appropriate column replaced
        let mut num_mat = DMatrix::<f64>::zeros(dim + 1, dim + 1);
        for i in 0..=dim {
            for j in 0..dim {
                if j == coord_idx {
                    num_mat[(i, j)] = mat[i][0]; // Replace with sum of squares
                } else if j < coord_idx {
                    num_mat[(i, j)] = mat[i][j + 1];
                } else {
                    // j > coord_idx
                    num_mat[(i, j)] = mat[i][j + 2];
                }
            }
            // Last column is always 1
            num_mat[(i, dim)] = 1.0;
        }

        center[coord_idx] = factor * num_mat.determinant();
        factor *= -1.0;
    }

    // Calculate radius
    let diff: Vec<f64> = center.iter()
        .zip(points[0].iter())
        .map(|(c, p)| c - p)
        .collect();
    let radius = fast_norm(&diff);

    Ok((center, radius))
}

/// Calculate volume of a simplex
pub fn volume(simplex: &[Vec<f64>]) -> f64 {
    let dim = simplex.len() - 1;
    let mut matrix = vec![vec![0.0; dim]; dim];

    let last = &simplex[dim];
    for i in 0..dim {
        for j in 0..dim {
            matrix[i][j] = simplex[i][j] - last[j];
        }
    }

    let det = fast_det(&matrix);
    let factorial = (1..=dim).product::<usize>() as f64;
    det.abs() / factorial
}

/// Calculate volume of a simplex in higher-dimensional embedding
pub fn simplex_volume_in_embedding(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    let num_verts = vertices.len();
    let dim = vertices[0].len();

    // Special case for triangles in 2D (Heron's formula)
    if dim == 2 && num_verts == 3 {
        let a = distance(&vertices[0], &vertices[1]);
        let b = distance(&vertices[1], &vertices[2]);
        let c = distance(&vertices[2], &vertices[0]);
        let s = 0.5 * (a + b + c);
        let area_sq = s * (s - a) * (s - b) * (s - c);

        if area_sq < 0.0 {
            if area_sq > -1e-15 {
                return Ok(0.0);
            }
            return Err(GeometryError::NumericalError(
                "Negative area squared".to_string()
            ));
        }
        return Ok(area_sq.sqrt());
    }

    // General case using Cayley-Menger determinant
    let n = num_verts;
    let mut bordered = DMatrix::<f64>::zeros(n + 1, n + 1);

    // Fill the bordered matrix
    bordered[(0, 0)] = 0.0;
    for i in 1..=n {
        bordered[(0, i)] = 1.0;
        bordered[(i, 0)] = 1.0;
    }

    for i in 0..n {
        for j in 0..n {
            if i == j {
                bordered[(i + 1, j + 1)] = 0.0;
            } else {
                let dist_sq = distance_squared(&vertices[i], &vertices[j]);
                bordered[(i + 1, j + 1)] = dist_sq;
            }
        }
    }

    let det = bordered.determinant();
    let factorial = (1..=num_verts-1).product::<usize>() as f64;
    let coeff = -((-2_i32).pow((num_verts - 1) as u32) as f64) * factorial * factorial;
    let vol_squared = det / coeff;

    if vol_squared < 0.0 {
        if vol_squared > -1e-15 {
            return Ok(0.0);
        }
        return Err(GeometryError::NumericalError(
            "Provided vertices do not form a simplex".to_string()
        ));
    }

    Ok(vol_squared.sqrt())
}

/// Calculate Euclidean distance between two points
fn distance(p1: &[f64], p2: &[f64]) -> f64 {
    let diff: Vec<f64> = p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| a - b)
        .collect();
    fast_norm(&diff)
}

/// Calculate squared Euclidean distance between two points
fn distance_squared(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}
