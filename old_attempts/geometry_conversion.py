import numpy as np
import trimesh
import argparse
import sys

def barycentric_interp(uv, tri_uv, tri_vertices):
    """
    Compute the barycentric interpolation of uv inside the triangle defined by tri_uv.
    Returns the interpolated 3D point if uv lies inside the triangle, otherwise None.
    """
    # Build 2x2 system to solve for barycentric weights (excluding the first weight)
    A = np.array([
        [tri_uv[1, 0] - tri_uv[0, 0], tri_uv[2, 0] - tri_uv[0, 0]],
        [tri_uv[1, 1] - tri_uv[0, 1], tri_uv[2, 1] - tri_uv[0, 1]]
    ])
    b = uv - tri_uv[0]
    try:
        lambdas = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    l1, l2 = lambdas
    l0 = 1.0 - l1 - l2
    # Check if the barycentric coordinates are within the triangle
    if l0 < -1e-5 or l1 < -1e-5 or l2 < -1e-5:
        return None  # Outside the triangle
    return l0 * tri_vertices[0] + l1 * tri_vertices[1] + l2 * tri_vertices[2]

def process_submesh(submesh, grid_size):
    """
    Process a single submesh to create a geometry image.
    Returns a tuple (geom_image, count_image) where:
      - geom_image is an array of shape (grid_size, grid_size, 3) with accumulated 3D points.
      - count_image counts contributions per pixel.
    If the submesh does not have UV mapping, returns None.
    """
    if not hasattr(submesh.visual, 'uv') or submesh.visual.uv is None:
        print("Warning: A submesh does not contain UV mapping. Skipping it.")
        return None

    vertices = submesh.vertices            # (num_vertices, 3)
    faces = submesh.faces                  # (num_faces, 3)
    uvs = submesh.visual.uv                # (num_vertices, 2)

    geom_image = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    count_image = np.zeros((grid_size, grid_size), dtype=np.int32)

    # For each pixel, compute its UV coordinate in [0,1]
    for i in range(grid_size):
        for j in range(grid_size):
            uv_coord = np.array([j / (grid_size - 1), i / (grid_size - 1)])
            found = False
            for face in faces:
                tri_uv = uvs[face]          # 3 x 2 array
                tri_vertices = vertices[face]  # 3 x 3 array
                point3D = barycentric_interp(uv_coord, tri_uv, tri_vertices)
                if point3D is not None:
                    geom_image[i, j] += point3D
                    count_image[i, j] += 1
                    found = True
                    break  # Only take the first hit for this submesh
            # Optionally, if not found, leave as zero.
    num_hits = np.count_nonzero(count_image)
    print("Number of pixels that got a valid 3D point:", num_hits)

    return geom_image, count_image

def obj_to_geometry_image(obj_path, grid_size=256, output_path='geometry_image.npy'):
    """
    Load an .obj file and create a geometry image.
    If the file loads as a scene with multiple submeshes (each with its own UV mapping),
    each submesh is processed separately and their contributions are merged.
    """
    mesh = trimesh.load(obj_path, process=False)
    # If the file loads as a scene, process each submesh separately.
    if isinstance(mesh, trimesh.Scene):
        submeshes = mesh.geometry.values()
    else:
        submeshes = [mesh]

    # Create arrays to accumulate geometry and counts
    final_geom_image = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    final_count_image = np.zeros((grid_size, grid_size), dtype=np.int32)

    for submesh in submeshes:
        result = process_submesh(submesh, grid_size)
        if result is None:
            continue
        geom_image, count_image = result
        # Accumulate the contributions
        final_geom_image += geom_image
        final_count_image += count_image

    # Average the contributions in pixels with one or more samples.
    nonzero = final_count_image > 0
    final_geom_image[nonzero] /= final_count_image[nonzero, None]

    np.save(output_path, final_geom_image)
    print(f"Geometry image saved to {output_path}")

def geometry_image_to_obj(geometry_image_path, grid_size=None, output_obj_path='reconstructed.obj'):
    """
    Load a geometry image (numpy .npy file) and convert it back to a 3D mesh.
    The geometry image is assumed to be a regular grid with shape (H, W, 3).
    The output mesh is created by treating each pixel as a vertex and connecting adjacent pixels.
    """
    geometry_image = np.load(geometry_image_path)
    
    if grid_size is None:
        # Use the geometry image dimensions
        grid_size = geometry_image.shape[0]
    
    # Flatten the grid to get a list of vertices.
    vertices = geometry_image.reshape(-1, 3)

    faces = []
    # Create faces (two triangles per grid cell)
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            idx_tl = i * grid_size + j
            idx_tr = i * grid_size + (j + 1)
            idx_bl = (i + 1) * grid_size + j
            idx_br = (i + 1) * grid_size + (j + 1)
            faces.append([idx_tl, idx_tr, idx_bl])
            faces.append([idx_bl, idx_tr, idx_br])
    
    faces = np.array(faces)
    
    # Create and export the mesh.
    mesh_reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_reconstructed.export(output_obj_path)
    print(f"Reconstructed .obj file saved to {output_obj_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert between .obj and geometry image representations.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Subparser for converting .obj to geometry image
    parser_to_img = subparsers.add_parser('to_image', help="Convert .obj to geometry image")
    parser_to_img.add_argument('obj_path', type=str, help="Path to the input .obj file")
    parser_to_img.add_argument('--grid_size', type=int, default=256, help="Resolution of the geometry image grid (default: 256)")
    parser_to_img.add_argument('--output', type=str, default='geometry_image.npy', help="Output file for the geometry image (default: geometry_image.npy)")
    
    # Subparser for converting geometry image back to .obj
    parser_to_obj = subparsers.add_parser('to_obj', help="Convert geometry image back to .obj")
    parser_to_obj.add_argument('geometry_image_path', type=str, help="Path to the input geometry image (.npy file)")
    parser_to_obj.add_argument('--grid_size', type=int, default=None, help="Grid size used to create the geometry image (if different from image dimensions)")
    parser_to_obj.add_argument('--output', type=str, default='reconstructed.obj', help="Output .obj file (default: reconstructed.obj)")
    
    args = parser.parse_args()
    
    if args.command == 'to_image':
        obj_to_geometry_image(args.obj_path, grid_size=args.grid_size, output_path=args.output)
    elif args.command == 'to_obj':
        geometry_image_to_obj(args.geometry_image_path, grid_size=args.grid_size, output_obj_path=args.output)

if __name__ == '__main__':
    main()
