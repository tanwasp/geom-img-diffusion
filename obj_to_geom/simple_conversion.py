import numpy as np
import math
import sys
from PIL import Image

def read_obj(filename):
    """
    Reads an .obj file and extracts vertices and triangular faces.
    Ignores normals, texture coordinates, material libraries, group definitions, and lines.
    """
    vertices = []
    faces = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Faces can have entries like "1//1"; we split and use the first part.
                face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(face) >= 3:
                    faces.append(face[:3])
    return np.array(vertices), np.array(faces)

def write_obj(filename, vertices, faces):
    """
    Writes vertices and faces to an .obj file.
    """
    with open(filename, "w") as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

def compute_centroid(vertices):
    """Compute the centroid of the mesh."""
    return np.mean(vertices, axis=0)

def spherical_uv(vertices):
    """
    Compute spherical UV coordinates for each vertex after centering the mesh.
    u = (atan2(z, x) / (2*pi)) + 0.5,
    v = acos(y / r) / pi.
    """
    # Center the vertices by subtracting the centroid
    centroid = compute_centroid(vertices)
    centered = vertices - centroid

    uvs = []
    for v in centered:
        x, y, z = v
        r = np.linalg.norm(v)
        if r == 0:
            u, v_coord = 0, 0
        else:
            theta = math.atan2(z, x)
            phi = math.acos(np.clip(y / r, -1.0, 1.0))
            u = (theta / (2 * math.pi)) + 0.5
            v_coord = phi / math.pi
        uvs.append([u, v_coord])
    return np.array(uvs)

def barycentric_coords(p, a, b, c):
    """Compute barycentric coordinates of point p with respect to triangle (a, b, c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return -1, -1, -1
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return u, v, w

def point_in_triangle(bary):
    """Return True if barycentric coordinates indicate the point is inside the triangle."""
    u, v, w = bary
    return (u >= 0) and (v >= 0) and (w >= 0)

def triangle_area(pts):
    """Compute the area of a triangle given by three 2D points."""
    a, b, c = pts
    return abs(0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))

def create_geometry_image(vertices, faces, uvs, resolution=256):
    """
    Rasterize the 3D mesh onto a geometry image.
    Uses weighted contributions (weighted by the area in UV space) when multiple triangles overlap.
    """
    geo_img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    weight_img = np.zeros((resolution, resolution), dtype=np.float32)

    # For each triangle face
    for face in faces:
        idx0, idx1, idx2 = face
        v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
        uv0, uv1, uv2 = uvs[idx0], uvs[idx1], uvs[idx2]

        # Map UVs [0,1] to pixel coordinates
        p0 = np.array([uv0[0] * (resolution - 1), uv0[1] * (resolution - 1)])
        p1 = np.array([uv1[0] * (resolution - 1), uv1[1] * (resolution - 1)])
        p2 = np.array([uv2[0] * (resolution - 1), uv2[1] * (resolution - 1)])

        # Compute area weight in UV space (smaller triangles get higher weight)
        area = triangle_area([p0, p1, p2])
        if area == 0:
            continue

        # Bounding box for the triangle
        min_x = int(max(min(p0[0], p1[0], p2[0]), 0))
        max_x = int(min(max(p0[0], p1[0], p2[0]), resolution - 1))
        min_y = int(max(min(p0[1], p1[1], p2[1]), 0))
        max_y = int(min(max(p0[1], p1[1], p2[1]), resolution - 1))

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                p = np.array([i, j])
                bary = barycentric_coords(p, p0, p1, p2)
                if point_in_triangle(bary):
                    # Compute weighted interpolation: weight factor is the inverse of the area.
                    weight = 1.0 / area
                    pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
                    geo_img[j, i] += pos * weight
                    weight_img[j, i] += weight

    # Normalize contributions where they exist.
    nonzero = weight_img > 0
    geo_img[nonzero] /= weight_img[nonzero, None]
    return geo_img

def geometry_image_to_obj(geo_img):
    """
    Reconstruct a mesh from the geometry image.
    Each pixel becomes a vertex and adjacent pixels form two triangles per grid cell.
    """
    h, w, _ = geo_img.shape
    vertices = []
    vertex_indices = np.zeros((h, w), dtype=int)

    idx = 0
    for i in range(h):
        for j in range(w):
            vertices.append(geo_img[i, j].tolist())
            vertex_indices[i, j] = idx
            idx += 1
    vertices = np.array(vertices)

    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = vertex_indices[i, j]
            v1 = vertex_indices[i, j + 1]
            v2 = vertex_indices[i + 1, j]
            v3 = vertex_indices[i + 1, j + 1]
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    faces = np.array(faces)
    return vertices, faces

def laplacian_smoothing(vertices, faces, iterations=5, lambda_factor=0.5):
    """
    Applies simple Laplacian smoothing to the mesh.
    For each vertex, moves it toward the average of its neighbors.
    """
    from collections import defaultdict
    neighbors = defaultdict(set)
    for face in faces:
        for i in range(3):
            a = face[i]
            b = face[(i+1)%3]
            c = face[(i+2)%3]
            neighbors[a].add(b)
            neighbors[a].add(c)
    
    vertices_smoothed = vertices.copy()
    for _ in range(iterations):
        new_vertices = vertices_smoothed.copy()
        for i in range(len(vertices_smoothed)):
            if len(neighbors[i]) == 0:
                continue
            avg = np.mean([vertices_smoothed[j] for j in neighbors[i]], axis=0)
            new_vertices[i] = vertices_smoothed[i] * (1 - lambda_factor) + avg * lambda_factor
        vertices_smoothed = new_vertices
    return vertices_smoothed

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py input.obj output_prefix [resolution]")
        sys.exit(1)
    input_obj = sys.argv[1]
    output_prefix = sys.argv[2]
    resolution = 256
    if len(sys.argv) > 3:
        resolution = int(sys.argv[3])
    
    # Read the .obj file
    vertices, faces = read_obj(input_obj)
    print("Read {} vertices and {} faces.".format(len(vertices), len(faces)))
    
    # Compute improved UV mapping (centering the mesh first)
    uvs = spherical_uv(vertices)
    
    # Create the geometry image using weighted rasterization
    geo_img = create_geometry_image(vertices, faces, uvs, resolution=resolution)
    print("Created geometry image of resolution {}x{}.".format(resolution, resolution))
    
    # Save the geometry image for visualization.
    min_val = geo_img.min()
    max_val = geo_img.max()
    norm_img = (255 * (geo_img - min_val) / (max_val - min_val)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(norm_img)
    image.save(output_prefix + "_geometry_image.png")
    print("Saved geometry image as {}_geometry_image.png".format(output_prefix))
    
    # Reconstruct the mesh from the geometry image.
    rec_vertices, rec_faces = geometry_image_to_obj(geo_img)
    print("Reconstructed mesh has {} vertices and {} faces.".format(len(rec_vertices), len(rec_faces)))
    
    # Apply Laplacian smoothing to the reconstructed mesh.
    rec_vertices_smoothed = laplacian_smoothing(rec_vertices, rec_faces, iterations=10, lambda_factor=0.4)
    out_obj = output_prefix + "_reconstructed.obj"
    write_obj(out_obj, rec_vertices_smoothed, rec_faces)
    print("Saved reconstructed (smoothed) .obj as {}.".format(out_obj))

if __name__ == "__main__":
    main()
