use std::collections::{HashSet, HashMap, VecDeque};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PySet, PyTuple};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use nalgebra::DMatrix;
use crate::geometry;

pub type PointIndex = usize;
pub type Simplex = Vec<PointIndex>;

#[derive(Debug, Clone)]
pub struct TriangulationCore {
    pub vertices: Vec<Vec<f64>>,
    pub simplices: HashSet<Simplex>,
    pub vertex_to_simplices: Vec<HashSet<Simplex>>,
    dim: usize,
}

impl TriangulationCore {
    /// Create a new triangulation from initial points
    pub fn new(coords: Vec<Vec<f64>>) -> Result<Self, String> {
        if coords.is_empty() {
            return Err("Please provide at least one simplex".to_string());
        }

        let dim = coords[0].len();
        if dim < 2 {
            return Err("Triangulation class only supports dim >= 2".to_string());
        }

        if coords.len() < dim + 1 {
            return Err("Please provide at least one simplex".to_string());
        }

        // Check all coordinates have same dimension
        for coord in &coords {
            if coord.len() != dim {
                return Err("Coordinates dimension mismatch".to_string());
            }
        }

        // Check that initial simplex is not degenerate
        let vectors: Vec<Vec<f64>> = coords[1..].iter()
            .map(|c| c.iter().zip(&coords[0]).map(|(a, b)| a - b).collect())
            .collect();

        if !is_full_rank(&vectors) {
            return Err("Initial simplex has zero volume (points are linearly dependent)".to_string());
        }

        let mut tri = TriangulationCore {
            vertices: coords.clone(),
            simplices: HashSet::new(),
            vertex_to_simplices: coords.iter().map(|_| HashSet::new()).collect(),
            dim,
        };

        // Use Delaunay-like initialization for the initial points
        tri.initialize_simplices()?;

        Ok(tri)
    }

    /// Initialize simplices using a simple approach
    fn initialize_simplices(&mut self) -> Result<(), String> {
        // For now, if we have exactly dim+1 points, create a single simplex
        // For more points, we'll need a proper Delaunay triangulation
        if self.vertices.len() == self.dim + 1 {
            let simplex: Simplex = (0..=self.dim).collect();
            self.add_simplex(simplex);
        } else {
            // Simple approach: create simplices by connecting points
            // This is a placeholder - in production we'd use a proper Delaunay algorithm
            self.delaunay_triangulation()?;
        }
        Ok(())
    }

    /// Simple Delaunay triangulation for initial points
    fn delaunay_triangulation(&mut self) -> Result<(), String> {
        // Create initial simplex and add remaining points incrementally
        if self.vertices.len() < self.dim + 1 {
            return Err("Not enough points for triangulation".to_string());
        }

        // Create the first simplex
        let initial_simplex: Simplex = (0..=self.dim).collect();
        self.add_simplex(initial_simplex);

        // Add remaining vertices one by one using Bowyer-Watson
        for i in (self.dim + 1)..self.vertices.len() {
            // Process each vertex already in the list
            let (_deleted, _added) = self.add_point_internal(i)?;
            // The simplices have already been updated by add_point_internal
        }

        Ok(())
    }

    /// Internal point addition for vertices already in the vertices list
    fn add_point_internal(&mut self, pt_index: usize) -> Result<(HashSet<Simplex>, HashSet<Simplex>), String> {
        let point = self.vertices[pt_index].clone();
        let containing = self.locate_point(&point);

        if containing.is_none() {
            // Point is outside, need to extend hull
            self.extend_hull_internal(pt_index)
        } else {
            // Point is inside, use normal Bowyer-Watson
            self.add_point_bowyer_watson(pt_index, containing)
        }
    }

    /// Internal version of extend_hull for points already in vertices
    fn extend_hull_internal(&mut self, pt_index: usize) -> Result<(HashSet<Simplex>, HashSet<Simplex>), String> {
        // Find hull faces
        let mut face_count: HashMap<Simplex, usize> = HashMap::new();

        for simplex in &self.simplices {
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                face.sort_unstable();
                *face_count.entry(face).or_insert(0) += 1;
            }
        }

        let hull_faces: Vec<Simplex> = face_count.into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(face, _)| face)
            .collect();

        // Compute center of convex hull
        let hull_center = if pt_index > 0 {
            let mut center = vec![0.0; self.dim];
            for vertex in &self.vertices[..pt_index] {
                for (i, &v) in vertex.iter().enumerate() {
                    center[i] += v;
                }
            }
            center.iter_mut().for_each(|c| *c /= pt_index as f64);
            center
        } else {
            vec![0.0; self.dim]
        };

        let new_vertex = self.vertices[pt_index].clone();
        let mut new_simplices = HashSet::new();

        // Create new simplices from visible hull faces
        for face in hull_faces {
            let face_vertices = self.get_vertices(&face);

            if self.is_face_visible(&face_vertices, &hull_center, &new_vertex) {
                let mut new_simplex = face;
                new_simplex.push(pt_index);
                new_simplex.sort_unstable();

                if !self.is_simplex_degenerate(&new_simplex) {
                    self.add_simplex(new_simplex.clone());
                    new_simplices.insert(new_simplex);
                }
            }
        }

        if new_simplices.is_empty() {
            return Err("Candidate vertex is inside the hull".to_string());
        }

        // Run Bowyer-Watson to fix any Delaunay violations
        self.add_point_bowyer_watson(pt_index, None)
    }

    /// Add a simplex to the triangulation
    pub fn add_simplex(&mut self, mut simplex: Simplex) {
        simplex.sort_unstable();
        self.simplices.insert(simplex.clone());
        for &vertex in &simplex {
            self.vertex_to_simplices[vertex].insert(simplex.clone());
        }
    }

    /// Delete a simplex from the triangulation
    pub fn delete_simplex(&mut self, mut simplex: Simplex) {
        simplex.sort_unstable();
        self.simplices.remove(&simplex);
        for &vertex in &simplex {
            self.vertex_to_simplices[vertex].remove(&simplex);
        }
    }

    /// Get vertices by indices
    pub fn get_vertices(&self, indices: &[PointIndex]) -> Vec<Vec<f64>> {
        indices.iter().map(|&i| self.vertices[i].clone()).collect()
    }

    /// Find which simplex contains a point
    pub fn locate_point(&self, point: &[f64]) -> Option<Simplex> {
        for simplex in &self.simplices {
            let vertices = self.get_vertices(simplex);
            if geometry::point_in_simplex(point, &vertices, 1e-8) {
                return Some(simplex.clone());
            }
        }
        None
    }

    /// Check if a point is in the circumcircle of a simplex
    pub fn point_in_circumcircle(&self, pt_index: PointIndex, simplex: &Simplex) -> bool {
        let vertices = self.get_vertices(simplex);
        let point = &self.vertices[pt_index];

        match geometry::circumsphere(&vertices) {
            Ok((center, radius)) => {
                let dist = geometry::fast_norm(&center.iter()
                    .zip(point.iter())
                    .map(|(c, p)| c - p)
                    .collect::<Vec<_>>());
                dist < radius * (1.0 + 1e-8)
            }
            Err(_) => false,
        }
    }

    /// Bowyer-Watson algorithm for adding a point
    pub fn add_point_bowyer_watson(
        &mut self,
        pt_index: PointIndex,
        containing_simplex: Option<Simplex>
    ) -> Result<(HashSet<Simplex>, HashSet<Simplex>), String> {
        let mut queue = VecDeque::new();
        let mut done_simplices = HashSet::new();
        let mut bad_triangles = HashSet::new();

        // Initialize queue
        if let Some(simplex) = containing_simplex {
            queue.push_back(simplex);
        } else {
            // Add all simplices connected to this vertex (if it already exists)
            if pt_index < self.vertex_to_simplices.len() {
                for simplex in &self.vertex_to_simplices[pt_index] {
                    queue.push_back(simplex.clone());
                }
            }
            // If no simplices found, try all simplices (expensive but necessary for new points)
            if queue.is_empty() {
                for simplex in &self.simplices {
                    queue.push_back(simplex.clone());
                }
            }
        }

        // Process queue
        while let Some(simplex) = queue.pop_front() {
            if done_simplices.contains(&simplex) {
                continue;
            }
            done_simplices.insert(simplex.clone());

            if self.point_in_circumcircle(pt_index, &simplex) {
                bad_triangles.insert(simplex.clone());

                // Get neighbors sharing a face with this simplex
                let neighbors = self.get_face_sharing_neighbors(&simplex);
                for neighbor in neighbors {
                    if !done_simplices.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Delete bad triangles
        for simplex in &bad_triangles {
            self.delete_simplex(simplex.clone());
        }

        // Create new triangles from the hole boundary
        let hole_faces = self.get_hole_boundary(&bad_triangles);
        let mut new_triangles = HashSet::new();

        for face in hole_faces {
            if !face.contains(&pt_index) {
                let mut new_simplex = face;
                new_simplex.push(pt_index);
                new_simplex.sort_unstable();

                // Check that the new simplex is not degenerate
                if !self.is_simplex_degenerate(&new_simplex) {
                    self.add_simplex(new_simplex.clone());
                    new_triangles.insert(new_simplex);
                }
            }
        }

        Ok((bad_triangles, new_triangles))
    }

    /// Get neighbors that share a face with the given simplex
    fn get_face_sharing_neighbors(&self, simplex: &Simplex) -> HashSet<Simplex> {
        let mut neighbors = HashSet::new();

        // A face is the simplex minus one vertex
        for i in 0..simplex.len() {
            let mut face = simplex.clone();
            face.remove(i);

            // Find all simplices containing this face
            let containing = self.get_simplices_containing_face(&face);
            for s in containing {
                if s != *simplex {
                    neighbors.insert(s);
                }
            }
        }

        neighbors
    }

    /// Get all simplices containing a given face
    fn get_simplices_containing_face(&self, face: &[PointIndex]) -> HashSet<Simplex> {
        if face.is_empty() {
            return HashSet::new();
        }

        // Start with simplices containing the first vertex
        let mut result = self.vertex_to_simplices[face[0]].clone();

        // Intersect with simplices containing other vertices
        for &vertex in &face[1..] {
            result = result.intersection(&self.vertex_to_simplices[vertex])
                .cloned()
                .collect();
        }

        // Filter to only those that actually contain the entire face
        result.into_iter()
            .filter(|simplex| face.iter().all(|v| simplex.contains(v)))
            .collect()
    }

    /// Get the boundary of the hole created by removing bad triangles
    fn get_hole_boundary(&self, bad_triangles: &HashSet<Simplex>) -> Vec<Simplex> {
        let mut face_count: HashMap<Simplex, usize> = HashMap::new();

        // Count how many times each face appears
        for simplex in bad_triangles {
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                face.sort_unstable();
                *face_count.entry(face).or_insert(0) += 1;
            }
        }

        // Boundary faces appear exactly once
        face_count.into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(face, _)| face)
            .collect()
    }

    /// Check if a simplex is degenerate (has zero volume)
    fn is_simplex_degenerate(&self, simplex: &Simplex) -> bool {
        let vertices = self.get_vertices(simplex);
        geometry::volume(&vertices) < 1e-15
    }

    /// Add a new point to the triangulation
    pub fn add_point(&mut self, point: Vec<f64>) -> Result<(HashSet<Simplex>, HashSet<Simplex>), String> {
        // Check if point already exists
        for (_i, vertex) in self.vertices.iter().enumerate() {
            if vertex.iter().zip(&point).all(|(a, b)| (a - b).abs() < 1e-10) {
                return Err("Point already in triangulation".to_string());
            }
        }

        // Find containing simplex
        let containing = self.locate_point(&point);

        // Add vertex
        let pt_index = self.vertices.len();
        self.vertices.push(point);
        self.vertex_to_simplices.push(HashSet::new());

        // If point is outside convex hull, we need to extend the hull
        if containing.is_none() {
            self.extend_hull(pt_index)
        } else {
            self.add_point_bowyer_watson(pt_index, containing)
        }
    }

    /// Extend the convex hull to include a new point
    fn extend_hull(&mut self, pt_index: PointIndex) -> Result<(HashSet<Simplex>, HashSet<Simplex>), String> {
        // Find hull faces (faces that belong to only one simplex)
        let mut face_count: HashMap<Simplex, usize> = HashMap::new();

        for simplex in &self.simplices {
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                face.sort_unstable();
                *face_count.entry(face).or_insert(0) += 1;
            }
        }

        let hull_faces: Vec<Simplex> = face_count.into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(face, _)| face)
            .collect();

        // Compute center of convex hull (roughly)
        let hull_center = if !self.vertices.is_empty() {
            let mut center = vec![0.0; self.dim];
            for vertex in &self.vertices[..self.vertices.len()-1] {  // Exclude the new point
                for (i, &v) in vertex.iter().enumerate() {
                    center[i] += v;
                }
            }
            center.iter_mut().for_each(|c| *c /= (self.vertices.len() - 1) as f64);
            center
        } else {
            vec![0.0; self.dim]
        };

        let new_vertex = self.vertices[pt_index].clone();
        let mut new_simplices = HashSet::new();

        // Create new simplices from visible hull faces
        for face in hull_faces {
            let face_vertices = self.get_vertices(&face);

            // Check orientation - is this face visible from the new point?
            if self.is_face_visible(&face_vertices, &hull_center, &new_vertex) {
                let mut new_simplex = face;
                new_simplex.push(pt_index);
                new_simplex.sort_unstable();

                if !self.is_simplex_degenerate(&new_simplex) {
                    self.add_simplex(new_simplex.clone());
                    new_simplices.insert(new_simplex);
                }
            }
        }

        if new_simplices.is_empty() {
            // Revert adding the vertex
            self.vertices.pop();
            self.vertex_to_simplices.pop();
            return Err("Candidate vertex is inside the hull".to_string());
        }

        // Run Bowyer-Watson to fix any Delaunay violations
        self.add_point_bowyer_watson(pt_index, None)
    }

    /// Check if a face is visible from a point
    fn is_face_visible(&self, face_vertices: &[Vec<f64>], inside_point: &[f64], outside_point: &[f64]) -> bool {
        // Compute orientation using determinant
        let orientation_inside = compute_orientation(face_vertices, inside_point);
        let orientation_outside = compute_orientation(face_vertices, outside_point);

        // If orientations are opposite, the face is visible
        orientation_inside * orientation_outside < 0.0
    }

    /// Get the convex hull vertices
    pub fn hull(&self) -> HashSet<PointIndex> {
        let mut face_count: HashMap<Simplex, usize> = HashMap::new();

        for simplex in &self.simplices {
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                face.sort_unstable();
                *face_count.entry(face).or_insert(0) += 1;
            }
        }

        let mut hull = HashSet::new();
        for (face, count) in face_count {
            if count == 1 {
                for vertex in face {
                    hull.insert(vertex);
                }
            }
        }

        hull
    }

    /// Calculate volume of a simplex
    pub fn volume(&self, simplex: &Simplex) -> f64 {
        let vertices = self.get_vertices(simplex);
        geometry::volume(&vertices)
    }
}

/// Check if a matrix has full rank
fn is_full_rank(vectors: &[Vec<f64>]) -> bool {
    if vectors.is_empty() {
        return false;
    }

    let n = vectors.len();
    let m = vectors[0].len();

    let mut matrix = DMatrix::<f64>::zeros(n, m);
    for (i, vec) in vectors.iter().enumerate() {
        for (j, &val) in vec.iter().enumerate() {
            matrix[(i, j)] = val;
        }
    }

    matrix.rank(1e-10) == n.min(m)
}

/// Compute orientation of a face with respect to a point
fn compute_orientation(face_vertices: &[Vec<f64>], point: &[f64]) -> f64 {
    let n = face_vertices.len();

    // Build matrix: subtract point from each face vertex
    let mut matrix = DMatrix::<f64>::zeros(n, n);

    for (i, vertex) in face_vertices.iter().enumerate() {
        for (j, &v) in vertex.iter().enumerate().take(n) {
            let p = point.get(j).unwrap_or(&0.0);
            matrix[(i, j)] = v - p;
        }
    }

    let det = matrix.determinant();

    // Return the sign of the determinant
    if det.abs() < 1e-15 {
        0.0
    } else if det > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Python-facing Triangulation class
#[pyclass]
pub struct RustTriangulation {
    core: TriangulationCore,
}

#[pymethods]
impl RustTriangulation {
    #[new]
    fn new(coords: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let shape = coords.shape();
        let n_points = shape[0];
        let dim = shape[1];

        // Convert numpy array to Vec<Vec<f64>>
        let mut points = Vec::new();
        let array = coords.as_array();
        for i in 0..n_points {
            let mut point = Vec::new();
            for j in 0..dim {
                point.push(array[[i, j]]);
            }
            points.push(point);
        }

        let core = TriangulationCore::new(points)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(RustTriangulation { core })
    }

    /// Add a point to the triangulation
    fn add_point(&mut self, py: Python, point: Vec<f64>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (deleted, added) = self.core.add_point(point)
            .map_err(|e| PyValueError::new_err(e))?;

        // Convert to Python sets of tuples
        let deleted_py = deleted.into_iter()
            .map(|s| PyTuple::new_bound(py, s).into())
            .collect::<Vec<Py<PyAny>>>();
        let added_py = added.into_iter()
            .map(|s| PyTuple::new_bound(py, s).into())
            .collect::<Vec<Py<PyAny>>>();

        let deleted_set = PySet::new_bound(py, &deleted_py)?;
        let added_set = PySet::new_bound(py, &added_py)?;

        Ok((deleted_set.into(), added_set.into()))
    }

    /// Get vertices
    fn get_vertices(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let vertices = &self.core.vertices;
        let array = PyArray2::from_vec2_bound(py, vertices)
            .map_err(|e| PyValueError::new_err(format!("Failed to create array: {:?}", e)))?;
        Ok(array.into())
    }

    /// Get simplices as a set
    fn get_simplices(&self, py: Python) -> PyResult<Py<PyAny>> {
        let simplices_py: Vec<Py<PyAny>> = self.core.simplices.iter()
            .map(|s| PyTuple::new_bound(py, s.clone()).into())
            .collect();

        let set = PySet::new_bound(py, &simplices_py)?;
        Ok(set.into())
    }

    /// Locate which simplex contains a point
    fn locate_point(&self, py: Python, point: Vec<f64>) -> PyResult<Py<PyAny>> {
        match self.core.locate_point(&point) {
            Some(simplex) => Ok(PyTuple::new_bound(py, simplex).into()),
            None => Ok(PyTuple::empty_bound(py).into()),
        }
    }

    /// Get hull vertices
    fn hull(&self, py: Python) -> PyResult<Py<PyAny>> {
        let hull = self.core.hull();
        let hull_py: Vec<usize> = hull.into_iter().collect();
        let set = PySet::new_bound(py, &hull_py)?;
        Ok(set.into())
    }

    /// Calculate volume of a simplex
    fn volume(&self, simplex: Vec<usize>) -> f64 {
        self.core.volume(&simplex)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.core.dim
    }
}
