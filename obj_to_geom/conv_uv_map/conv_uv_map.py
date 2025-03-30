import numpy as np
from PIL import Image
import trimesh
import argparse

def load_obj(file_path):
    """
    Load an OBJ file using trimesh.
    If a Scene is returned, merge all geometry into one mesh.
    """
    mesh_or_scene = trimesh.load(file_path, process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        # Merge all geometry into a single mesh.
        mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        mesh = mesh_or_scene
    return mesh


def barycentric_weights(tri, p):
    """
    Compute barycentric weights for point p (2,) with respect to triangle tri (3,2).
    Returns a numpy array [w0, w1, w2]. If the triangle is degenerate, returns negative weights.
    """
    # Set up the linear system:
    # p = w0 * v0 + w1 * v1 + w2 * v2, with w2 = 1 - w0 - w1.
    A = np.array([
        [tri[0,0] - tri[2,0], tri[1,0] - tri[2,0]],
        [tri[0,1] - tri[2,1], tri[1,1] - tri[2,1]]
    ])
    b = np.array([p[0]-tri[2,0], p[1]-tri[2,1]])
    try:
        sol = np.linalg.solve(A, b)
        w0, w1 = sol
        w2 = 1 - w0 - w1
        return np.array([w0, w1, w2])
    except np.linalg.LinAlgError:
        return np.array([-1, -1, -1])

def create_geometry_image(mesh, resolution=256):
    """
    Create a geometry image from the mesh.
    The mesh is assumed to have UV coordinates; if not, a simple planar projection is used.
    For each face, we rasterize the triangle in UV space and use barycentric interpolation to assign 3D vertex positions.
    """
    # Use provided UV mapping or compute a simple planar mapping from the x-y coordinates.
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = mesh.visual.uv
    else:
        # Compute UVs by normalizing the first two coordinates of the vertices
        verts2d = mesh.vertices[:, :2]
        uv = (verts2d - verts2d.min(axis=0)) / (verts2d.max(axis=0) - verts2d.min(axis=0))
    
    # Initialize an empty geometry image (float32 image storing [x,y,z] per pixel)
    geom_img = np.zeros((resolution, resolution, 3), dtype=np.float32)

    # For each face, get its three vertices and corresponding uv coordinates.
    for face in mesh.faces:
        # Get uv coordinates for the three vertices (shape: 3x2)
        uv_face = uv[face]
        # Convert UVs to image pixel coordinates (assume UV range [0,1])
        uv_pixels = uv_face * (resolution - 1)
        # Get the 3D vertex positions for the face (shape: 3x3)
        verts = mesh.vertices[face]

        # Compute bounding box of the triangle in pixel space
        min_x = int(np.floor(np.min(uv_pixels[:, 0])))
        max_x = int(np.ceil(np.max(uv_pixels[:, 0])))
        min_y = int(np.floor(np.min(uv_pixels[:, 1])))
        max_y = int(np.ceil(np.max(uv_pixels[:, 1])))

        # Loop over pixels in the bounding box
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                p = np.array([i, j])
                # Compute barycentric weights for p in the triangle defined by uv_pixels.
                w = barycentric_weights(uv_pixels, p)
                # Check if the point lies inside the triangle.
                if np.all(w >= 0):
                    # Interpolate the 3D position using the barycentrics.
                    pos = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
                    # Note: image coordinate (j,i): j is row (y) and i is column (x)
                    geom_img[j, i, :] = pos

    # For saving as a PNG we need to convert float data to 8-bit.
    # We linearly scale the geometry values based on the min and max across the image.
    min_val = geom_img.min()
    max_val = geom_img.max()
    geom_img_scaled = (geom_img - min_val) / (max_val - min_val) * 255
    geom_img_scaled = geom_img_scaled.astype(np.uint8)
    return geom_img_scaled, (min_val, max_val)

def reconstruct_obj_from_geometry_image(geom_img, scale_params, resolution=256):
    """
    Reconstruct a mesh from a geometry image.
    Each pixel becomes a vertex (after reversing the scale) and faces are generated as a grid.
    """
    min_val, max_val = scale_params
    # Reverse the scaling to get back to the original float values.
    geom_float = geom_img.astype(np.float32) / 255 * (max_val - min_val) + min_val
    # Each pixel is a vertex; reshape the image to a list of vertices.
    vertices = geom_float.reshape(-1, 3)
    faces = []

    # Create faces by connecting vertices in the grid (two triangles per grid cell)
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            idx_right = idx + 1
            idx_down = idx + resolution
            idx_down_right = idx + resolution + 1
            faces.append([idx, idx_down, idx_right])
            faces.append([idx_right, idx_down, idx_down_right])
    faces = np.array(faces)
    return vertices, faces

def write_obj(file_path, vertices, faces):
    """
    Write the vertices and faces to an OBJ file.
    """
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            # OBJ indices are 1-based.
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

    # Load the input OBJ mesh.
    mesh = load_obj(args.input_obj)
    # Create the geometry image.
    geom_img, scale_params = create_geometry_image(mesh, args.resolution)
    # Save the geometry image.
    Image.fromarray(geom_img).save(args.output_geo)
    print("Geometry image saved to", args.output_geo)

    # For the reconstruction, we simulate reading back the geometry image.
    geom_img_recon = np.array(Image.open(args.output_geo))
    vertices, faces = reconstruct_obj_from_geometry_image(geom_img_recon, scale_params, args.resolution)
    write_obj(args.output_obj, vertices, faces)
    print("Reconstructed OBJ saved to", args.output_obj)

if __name__ == "__main__":
    main()
