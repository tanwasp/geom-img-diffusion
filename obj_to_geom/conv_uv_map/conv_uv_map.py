import numpy as np
from PIL import Image
import trimesh
import argparse

def debug_vertex_stats(vertices, label="Vertices"):
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    print(f"DEBUG {label} Stats:")
    print("  Min: ", min_coords)
    print("  Max: ", max_coords)
    print("  Range: ", max_coords - min_coords)
    center = (min_coords + max_coords) / 2.0
    scale = (max_coords - min_coords).max()
    print("  Center: ", center)
    print("  Scale: ", scale)
    return center, scale

def debug_uv_stats(uvs):
    if uvs.size == 0:
        print("DEBUG: No UV coordinates found.")
        return None, None, None
    min_uv = uvs.min(axis=0)
    max_uv = uvs.max(axis=0)
    print("DEBUG UV Stats:")
    print("  Shape: ", uvs.shape)
    print("  First 5 UVs:\n", uvs[:5])
    print("  Min: ", min_uv)
    print("  Max: ", max_uv)
    print("  Range: ", max_uv - min_uv)
    unique_uv = np.unique(uvs, axis=0)
    print("DEBUG: Number of unique UVs:", unique_uv.shape[0])
    return min_uv, max_uv, unique_uv

def load_obj(file_path):
    mesh_or_scene = trimesh.load(file_path, process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        mesh = mesh_or_scene
    print("DEBUG: Loaded mesh with {} vertices and {} faces".format(len(mesh.vertices), len(mesh.faces)))
    return mesh

def barycentric_weights(tri, p):
    # Solve for barycentrics such that p = w0*v0 + w1*v1 + (1-w0-w1)*v2
    A = np.array([
        [tri[0,0] - tri[2,0], tri[1,0] - tri[2,0]],
        [tri[0,1] - tri[2,1], tri[1,1] - tri[2,1]]
    ])
    b = np.array([p[0] - tri[2,0], p[1] - tri[2,1]])
    try:
        sol = np.linalg.solve(A, b)
        w0, w1 = sol
        w2 = 1 - w0 - w1
        return np.array([w0, w1, w2])
    except np.linalg.LinAlgError:
        return np.array([-1, -1, -1])

def compute_pca_uv(mesh):
    """
    Compute UV coordinates based on a PCA projection.
    This finds the best-fit plane for the mesh vertices using SVD,
    then projects each vertex onto the first two principal directions.
    The UVs are then normalized to the [0,1] range.
    """
    vertices = mesh.vertices
    center = np.mean(vertices, axis=0)
    centered = vertices - center
    # SVD: the first two columns of Vt give the dominant directions.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Use the first two principal directions.
    basis = Vt[:2, :]  # shape: (2,3)
    uv = centered.dot(basis.T)  # shape: (n,2)
    # Normalize the UV coordinates to [0,1]
    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    uv_norm = (uv - min_uv) / (max_uv - min_uv)
    print("DEBUG: PCA-based UV Stats:")
    debug_uv_stats(uv_norm)
    return uv_norm

def create_geometry_image(mesh, resolution=256):
    # Debug the original mesh vertices.
    center, scale = debug_vertex_stats(mesh.vertices, label="Original Mesh Vertices")
    
    # Attempt to use the provided UV mapping.
    uv = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        if hasattr(mesh.visual, 'uv_index') and mesh.visual.uv_index is not None:
            uv = mesh.visual.uv[mesh.visual.uv_index]
            print("DEBUG: Using mesh UV coordinates with uv_index. UV shape:", uv.shape)
        else:
            uv = mesh.visual.uv
            print("DEBUG: Using mesh UV coordinates. UV shape:", uv.shape)
    else:
        print("DEBUG: No UVs found.")

    # If we have UVs, check for degeneracy.
    if uv is not None:
        min_uv, max_uv, unique_uv = debug_uv_stats(uv)
        uv_range = max_uv - min_uv if min_uv is not None and max_uv is not None else np.array([0, 0])
        if np.any(uv_range < 1e-3) or (unique_uv is not None and unique_uv.shape[0] <= 1):
            print("WARNING: Provided UV mapping is degenerate.")
            uv = None

    # If no valid UV mapping is available, compute our own using PCA.
    if uv is None:
        print("DEBUG: Falling back to computing UV mapping using PCA-based projection.")
        uv = compute_pca_uv(mesh)
    
    # Initialize the geometry image (storing float positions).
    geom_img = np.zeros((resolution, resolution, 3), dtype=np.float32)

    # Rasterize each face into the geometry image.
    for face in mesh.faces:
        uv_face = uv[face]              # shape (3,2)
        uv_pixels = uv_face * (resolution - 1)  # Scale UVs to pixel coordinates.
        verts = mesh.vertices[face]      # shape (3,3)

        # Compute the bounding box in pixel space.
        min_x = int(np.floor(np.min(uv_pixels[:, 0])))
        max_x = int(np.ceil(np.max(uv_pixels[:, 0])))
        min_y = int(np.floor(np.min(uv_pixels[:, 1])))
        max_y = int(np.ceil(np.max(uv_pixels[:, 1])))

        # Compute triangle area in UV space.
        tri_area = 0.5 * abs(np.linalg.det(np.array([
            [uv_face[1,0] - uv_face[0,0], uv_face[2,0] - uv_face[0,0]],
            [uv_face[1,1] - uv_face[0,1], uv_face[2,1] - uv_face[0,1]]
        ])))
        if tri_area < 1e-6:
            # print("DEBUG: Skipping degenerate face with area:", tri_area)
            continue

        # Loop over pixels in the bounding box.
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                p = np.array([i, j])
                w = barycentric_weights(uv_pixels, p)
                if np.all(w >= 0):
                    pos = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
                    geom_img[j, i, :] = pos

    # Scale the geometry image to 8-bit values for saving.
    min_val = geom_img.min()
    max_val = geom_img.max()
    print("DEBUG Geometry Image Stats:")
    print("  Min value:", min_val)
    print("  Max value:", max_val)
    if max_val - min_val == 0:
        print("WARNING: Geometry image values are all zero. Something went wrong during rasterization!")
        geom_img_scaled = np.zeros_like(geom_img, dtype=np.uint8)
    else:
        geom_img_scaled = (geom_img - min_val) / (max_val - min_val) * 255
        geom_img_scaled = geom_img_scaled.astype(np.uint8)
    return geom_img_scaled, (min_val, max_val)

def reconstruct_obj_from_geometry_image(geom_img, scale_params, resolution=256):
    min_val, max_val = scale_params
    # Reverse scaling from 8-bit back to float values.
    geom_float = geom_img.astype(np.float32) / 255 * (max_val - min_val) + min_val
    vertices = geom_float.reshape(-1, 3)
    faces = []
    # Create grid connectivity (two triangles per cell)
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            idx_right = idx + 1
            idx_down = idx + resolution
            idx_down_right = idx + resolution + 1
            faces.append([idx, idx_down, idx_right])
            faces.append([idx_right, idx_down, idx_down_right])
    faces = np.array(faces)
    print("DEBUG: Reconstructed mesh with {} vertices and {} faces".format(len(vertices), len(faces)))
    return vertices, faces

def write_obj(file_path, vertices, faces):
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            # OBJ file indices are 1-based.
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

def main():
    parser = argparse.ArgumentParser(
        description="Convert a .obj file into a geometry image and reconstruct a new .obj from the image."
    )
    parser.add_argument("input_obj", type=str, help="Input .obj file")
    parser.add_argument("output_geo", type=str, help="Output geometry image PNG file")
    parser.add_argument("output_obj", type=str, help="Output reconstructed .obj file")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the geometry image")
    args = parser.parse_args()

    mesh = load_obj(args.input_obj)
    geom_img, scale_params = create_geometry_image(mesh, args.resolution)
    Image.fromarray(geom_img).save(args.output_geo)
    print("DEBUG: Geometry image saved to", args.output_geo)

    geom_img_recon = np.array(Image.open(args.output_geo))
    vertices, faces = reconstruct_obj_from_geometry_image(geom_img_recon, scale_params, args.resolution)
    write_obj(args.output_obj, vertices, faces)
    print("DEBUG: Reconstructed OBJ saved to", args.output_obj)

if __name__ == "__main__":
    main()
