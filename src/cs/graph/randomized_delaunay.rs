use rand::seq::SliceRandom;
use rand::thread_rng;

pub type Point = (f64, f64);
pub type Triangle = (usize, usize, usize);

/// Computes the circumcircle of the triangle defined by points a, b, and c.
/// Returns Some((center, radius)) if the points are non-collinear, or None otherwise.
fn circumcircle(a: &Point, b: &Point, c: &Point) -> Option<((f64, f64), f64)> {
    let d = 2.0 * (a.0 * (b.1 - c.1) +
                   b.0 * (c.1 - a.1) +
                   c.0 * (a.1 - b.1));
    if d.abs() < 1e-6 {
        return None;
    }
    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    let center_x = (a2 * (b.1 - c.1) +
                    b2 * (c.1 - a.1) +
                    c2 * (a.1 - b.1)) / d;
    let center_y = (a2 * (c.0 - b.0) +
                    b2 * (a.0 - c.0) +
                    c2 * (b.0 - a.0)) / d;
    let center = (center_x, center_y);
    let radius = ((a.0 - center_x).powi(2) + (a.1 - center_y).powi(2)).sqrt();
    Some((center, radius))
}

/// Returns true if point p lies within the circle defined by center and radius.
fn point_in_circle(p: &Point, center: &(f64, f64), radius: f64) -> bool {
    let dx = p.0 - center.0;
    let dy = p.1 - center.1;
    (dx*dx + dy*dy).sqrt() <= radius + 1e-6
}

/// Performs a randomized incremental Delaunay Triangulation on the given set of points.
/// Returns a vector of triangles, where each triangle is represented as a triple of indices into the points vector.
/// Note: This is a simplified implementation and may not handle all degenerate cases.
pub fn randomized_delaunay(points: &[Point]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }
    
    // Compute bounding box.
    let (mut min_x, mut max_x) = (points[0].0, points[0].0);
    let (mut min_y, mut max_y) = (points[0].1, points[0].1);
    for &(x, y) in points.iter() {
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let s = dx.max(dy);
    
    // Create a "super triangle" that contains all the points.
    // These vertices are chosen to be well outside the bounding box.
    let super_a = (min_x - s, min_y - s);
    let super_b = (min_x - s, max_y + 2.0 * s);
    let super_c = (max_x + 2.0 * s, min_y - s);
    
    // Build an augmented points vector.
    let mut pts = points.to_vec();
    let n = pts.len();
    pts.push(super_a);
    pts.push(super_b);
    pts.push(super_c);
    
    // Initialize triangulation with the super triangle.
    let mut triangles: Vec<Triangle> = vec![(n, n + 1, n + 2)];
    
    // Randomize order of point indices from 0 to n-1.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);
    
    // Incrementally add each point.
    for &p_index in indices.iter() {
        let p = pts[p_index];
        let mut bad_triangles = Vec::new();
        // Find triangles whose circumcircle contains point p.
        for (i, &tri) in triangles.iter().enumerate() {
            let (i1, i2, i3) = tri;
            let a = pts[i1];
            let b = pts[i2];
            let c = pts[i3];
            if let Some((center, radius)) = circumcircle(&a, &b, &c) {
                if point_in_circle(&p, &center, radius) {
                    bad_triangles.push(i);
                }
            }
        }
        
        // Find boundary edges (edges that are not shared by two bad triangles).
        let mut edges = Vec::new();
        for &tri_index in bad_triangles.iter() {
            let tri = triangles[tri_index];
            let tri_edges = [(tri.0, tri.1), (tri.1, tri.2), (tri.2, tri.0)];
            for edge in tri_edges.iter() {
                // Count how many times this edge appears in bad triangles.
                let mut shared = false;
                for &other_index in bad_triangles.iter() {
                    if other_index == tri_index { continue; }
                    let other = triangles[other_index];
                    let other_edges = [(other.0, other.1), (other.1, other.2), (other.2, other.0)];
                    // An edge is the same if its vertices match regardless of order.
                    if other_edges.iter().any(|&(u, v)| {
                        (u == edge.0 && v == edge.1) || (u == edge.1 && v == edge.0)
                    }) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    edges.push(*edge);
                }
            }
        }
        
        // Remove bad triangles.
        // Remove indices in reverse order to avoid shifting.
        bad_triangles.sort_unstable_by(|a, b| b.cmp(a));
        for index in bad_triangles {
            triangles.remove(index);
        }
        
        // Re-triangulate the cavity with new triangles from the boundary edges.
        for edge in edges.iter() {
            triangles.push((edge.0, edge.1, p_index));
        }
    }
    
    // Remove triangles that contain vertices from the super triangle.
    triangles.retain(|&(i1, i2, i3)| {
        i1 < n && i2 < n && i3 < n
    });
    
    triangles
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_randomized_delaunay_simple() {
        // Define a simple set of 4 points forming a square.
        let points = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ];
        let tris = randomized_delaunay(&points);
        // For a square, there should be 2 triangles.
        assert_eq!(tris.len(), 2);
        // The triangles should reference indices within 0..4.
        for &(a, b, c) in tris.iter() {
            assert!(a < 4 && b < 4 && c < 4);
        }
    }
}
