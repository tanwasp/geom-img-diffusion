import numpy as np
from PIL import Image
import trimesh
import argparse
import math

def debug_vertex_stats(vertices, label="Vertices"):
    if vertices.size == 0:
        print(f"DEBUG {label} Stats: No vertices found.")
        return np.array([0,0,0]), 1.0 # Default center and scale

    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    print(f"DEBUG {label} Stats:")
    print("  Min: ", min_coords)
    print("  Max: ", max_coords)
    print("  Range: ", max_coords - min_coords)
    center = (min_coords + max_coords) / 2.0
    scale = (max_coords - min_coords).max()
    if scale < 1e-6: # Avoid division by zero or tiny scales
        scale = 1.0
    print("  Center: ", center)
    print("  Scale: ", scale)
    return center, scale

def debug_uv_stats(uvs, label="UV"):
    if uvs is None or uvs.size == 0:
        print(f"DEBUG {label} Stats: No UV coordinates found.")
        return None, None, None
    min_uv = uvs.min(axis=0)
    max_uv = uvs.max(axis=0)
    print(f"DEBUG {label} Stats:")
    print("  Shape: ", uvs.shape)
    # print("  First 5 UVs:\n", uvs[:5]) # Can be verbose
    print("  Min: ", min_uv)
    print("  Max: ", max_uv)
    print("  Range: ", max_uv - min_uv)
    try:
        # This can be slow for large meshes, use cautiously
        unique_uv = np.unique(uvs, axis=0)
        print(f"DEBUG {label}: Number of unique UVs:", unique_uv.shape[0])
        return min_uv, max_uv, unique_uv
    except Exception as e:
        print(f"DEBUG {label}: Error getting unique UVs: {e}")
        # Estimate uniqueness roughly
        unique_count_estimate = min(len(uvs), 1000) # Limit check size
        unique_uv_sample = np.unique(uvs[:unique_count_estimate], axis=0)
        print(f"DEBUG {label}: Approx unique UVs in sample:", unique_uv_sample.shape[0])
        return min_uv, max_uv, None # Indicate unique check failed or was partial


def load_obj_parts(file_path):
    """Loads an OBJ file and returns a list of individual mesh components."""
    try:
        mesh_or_scene = trimesh.load(file_path, process=False, split_object=True, group_material=True)
    except Exception as e:
        print(f"ERROR loading {file_path} with split/group options: {e}")
        print("Attempting simpler load...")
        try:
            mesh_or_scene = trimesh.load(file_path, process=False)
        except Exception as e2:
            print(f"ERROR: Failed simple load as well: {e2}")
            return [] # Return empty list on failure

    meshes = []
    if isinstance(mesh_or_scene, trimesh.Scene):
        print(f"DEBUG: Loaded Scene with {len(mesh_or_scene.geometry)} geometries.")
        # Iterate through the geometries in the scene
        for geom_name, geom in mesh_or_scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0 and len(geom.vertices) > 0:
                print(f"  - Processing geometry '{geom_name}' with {len(geom.vertices)} vertices, {len(geom.faces)} faces.")
                meshes.append(geom)
            # else: # Optional: handle point clouds or empty meshes if needed
            #     print(f"  - Skipping non-Trimesh or empty geometry '{geom_name}'")

    elif isinstance(mesh_or_scene, trimesh.Trimesh):
         if len(mesh_or_scene.faces) > 0 and len(mesh_or_scene.vertices) > 0:
            print("DEBUG: Loaded a single Trimesh.")
            meshes.append(mesh_or_scene)
         else:
             print("DEBUG: Loaded a single Trimesh but it is empty.")
    else:
        print(f"DEBUG: Loaded object is of type {type(mesh_or_scene)}, attempting to extract meshes if possible.")
        # Add logic here if other types like lists of meshes are returned by trimesh load
        if isinstance(mesh_or_scene, (list, tuple)):
             for item in mesh_or_scene:
                 if isinstance(item, trimesh.Trimesh) and len(item.faces) > 0 and len(item.vertices) > 0:
                     meshes.append(item)


    # Sometimes splitting results in many tiny meshes, optionally filter them
    # min_faces_threshold = 10
    # filtered_meshes = [m for m in meshes if len(m.faces) >= min_faces_threshold]
    # if len(filtered_meshes) < len(meshes):
    #     print(f"DEBUG: Filtered out {len(meshes) - len(filtered_meshes)} meshes with < {min_faces_threshold} faces.")
    #     meshes = filtered_meshes

    if not meshes:
        print("WARNING: No valid mesh components found after loading.")

    return meshes


def barycentric_weights(tri, p):
    # Check if triangle points are co-linear (or very close) for 2D UVs
    # Using cross product magnitude: area = 0.5 * |(v1-v0) x (v2-v0)|
    v0, v1, v2 = tri[0], tri[1], tri[2]
    area_vec = np.cross(v1 - v0, v2 - v0)
    # area = 0.5 * np.linalg.norm(area_vec) # If using 3D points
    # For 2D points, we can use the determinant formula used elsewhere
    area_det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])

    if abs(area_det) < 1e-9: # If triangle area is near zero in UV space
         # print("DEBUG: Degenerate triangle in UV space for barycentric weights.")
         return np.array([-1.0, -1.0, -1.0]) # Indicate failure


    # Solve for barycentrics such that p = w0*v0 + w1*v1 + w2*v2, where w0+w1+w2=1
    # Using the formula derived from vector algebra:
    # p = v2 + w0*(v0-v2) + w1*(v1-v2)
    # (p - v2) = w0*(v0-v2) + w1*(v1-v2)
    # Let vec_p = p-v2, vec_0 = v0-v2, vec_1 = v1-v2
    # We need to solve [vec_0, vec_1] * [w0, w1]^T = vec_p
    # This is equivalent to the matrix A used before:
    A = np.array([
        [tri[0,0] - tri[2,0], tri[1,0] - tri[2,0]],
        [tri[0,1] - tri[2,1], tri[1,1] - tri[2,1]]
    ])
    b = np.array([p[0] - tri[2,0], p[1] - tri[2,1]])

    try:
        # Use lstsq for potentially better stability than solve, especially near singular
        sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # Check if solution is valid (lstsq might return solution even for singular)
        if rank < 2:
             # print("DEBUG: Matrix A is rank deficient in barycentric calculation.")
             return np.array([-1.0, -1.0, -1.0])

        w0, w1 = sol
        w2 = 1.0 - w0 - w1
        # Clamp weights slightly outside [0, 1] to handle floating point inaccuracies at edges
        epsilon = 1e-5
        if w0 >= -epsilon and w1 >= -epsilon and w2 >= -epsilon:
             # Return clamped weights to avoid issues from minor errors
            return np.clip(np.array([w0, w1, w2]), 0.0, 1.0)
        else:
            return np.array([-1.0, -1.0, -1.0]) # Point outside triangle

    except np.linalg.LinAlgError:
        # print("DEBUG: LinAlgError during barycentric calculation.")
        return np.array([-1.0, -1.0, -1.0])


def compute_pca_uv(mesh):
    """Compute UV coordinates based on PCA projection for a *single* mesh."""
    if mesh.vertices.shape[0] < 3:
         print("WARNING: Cannot compute PCA UVs for mesh with < 3 vertices.")
         # Return dummy UVs matching vertex count, but mark as invalid?
         # Returning zeros might cause issues. Let's return None.
         return None

    vertices = mesh.vertices
    center = np.mean(vertices, axis=0)
    centered = vertices - center

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        if Vt.shape[0] < 2:
             print("WARNING: SVD resulted in fewer than 2 principal components.")
             return None # Not enough dimensions

        basis = Vt[:2, :]  # shape: (2,3)
        uv = centered.dot(basis.T)  # shape: (n,2)

        # Normalize the UV coordinates to [0,1]
        min_uv = uv.min(axis=0)
        max_uv = uv.max(axis=0)
        uv_range = max_uv - min_uv

        # Handle cases where one dimension might have zero range (e.g., flat plane)
        # Add a small epsilon to avoid division by zero
        uv_range[uv_range < 1e-6] = 1.0

        uv_norm = (uv - min_uv) / uv_range

        print("DEBUG: PCA-based UV Stats (normalized):")
        debug_uv_stats(uv_norm, "PCA UV")
        return uv_norm

    except np.linalg.LinAlgError:
        print("ERROR: SVD failed during PCA UV computation.")
        return None


def get_or_compute_uv_for_mesh(mesh, mesh_index):
    """ Tries to get existing valid UVs, otherwise computes PCA UVs for a single mesh. """
    uv = None
    print(f"--- Processing UVs for mesh component {mesh_index} ---")
    debug_vertex_stats(mesh.vertices, f"Mesh {mesh_index} Vertices")

    # Check for vertex texture coordinates (vt) associated directly with faces
    # This is often how UVs are stored in OBJ rather than mesh.visual.uv
    if hasattr(mesh, 'face_attributes') and 'uv' in mesh.face_attributes:
        # This structure might store UVs per face-vertex corner. Needs reshaping.
        # Example: shape might be (n_faces, 3, 2) -> reshape to (n_faces * 3, 2)
        # And then map back to unique vertices? This is complex.
        # Let's prioritize mesh.visual.uv first as it's simpler via trimesh.
        print(f"DEBUG Mesh {mesh_index}: Found 'uv' in face_attributes, but prioritizing mesh.visual.uv for simplicity.")
        # Potential future work: Properly handle face_attributes['uv']

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and mesh.visual.uv.size > 0:
        uv_raw = mesh.visual.uv
        print(f"DEBUG Mesh {mesh_index}: Found mesh.visual.uv with shape: {uv_raw.shape}")

        # Check if UVs are per-vertex or need indexing (less common with process=False, but check)
        # If uv shape matches vertex count, assume per-vertex UVs
        if uv_raw.shape[0] == mesh.vertices.shape[0]:
            uv = uv_raw
            print(f"DEBUG Mesh {mesh_index}: Assuming per-vertex UV mapping.")
        else:
             # If not per-vertex, trimesh might store indices differently.
             # This case is less likely with process=False, might indicate complex mapping.
             # For now, we'll try to use it but warn if it looks wrong.
             print(f"WARNING Mesh {mesh_index}: mesh.visual.uv shape {uv_raw.shape} doesn't match vertex count {mesh.vertices.shape[0]}. Using as is, but parameterization might be incorrect.")
             # We might need a mapping from faces to these UVs if they are texture coordinates (vt)
             # This structure isn't guaranteed by trimesh loading without processing.
             # Let's fall back to PCA if this looks suspicious.
             uv = None # Fallback to PCA if unsure about the mapping
             print(f"DEBUG Mesh {mesh_index}: Falling back to PCA due to ambiguous UV mapping.")


        if uv is not None:
            min_uv, max_uv, unique_uv = debug_uv_stats(uv, f"Mesh {mesh_index} Original UV")
            # Check for validity / degeneracy
            if min_uv is None or max_uv is None: # Debug stats indicate no UVs found
                print(f"WARNING Mesh {mesh_index}: Original UVs reported as non-existent by debug_uv_stats.")
                uv = None
            else:
                uv_range = max_uv - min_uv
                # Check if range is too small or if there's only one unique UV coord
                is_degenerate = np.any(uv_range < 1e-4) or (unique_uv is not None and unique_uv.shape[0] <= 1)
                # Also check if UVs are outside a reasonable range (e.g., [-10, 10]), indicating potential issues
                is_reasonable_range = np.all(min_uv > -10) and np.all(max_uv < 10)
                if is_degenerate:
                    print(f"WARNING Mesh {mesh_index}: Provided UV mapping is degenerate (range: {uv_range}, unique count: {unique_uv.shape[0] if unique_uv is not None else 'N/A'}).")
                    uv = None
                elif not is_reasonable_range:
                     print(f"WARNING Mesh {mesh_index}: Provided UV mapping has potentially problematic range (min: {min_uv}, max: {max_uv}).")
                     # Optional: Try normalizing existing UVs if range is just large
                     print(f"Attempting to normalize existing UVs for Mesh {mesh_index}.")
                     uv = (uv - min_uv) / uv_range
                     # Re-check stats after normalization
                     debug_uv_stats(uv, f"Mesh {mesh_index} Normalized Original UV")
                     # Check again if normalization made it valid [0,1]
                     if not (np.all(uv >= -1e-5) and np.all(uv <= 1.0 + 1e-5)):
                         print(f"WARNING Mesh {mesh_index}: Normalization failed to bring UVs into [0,1] range cleanly. Falling back.")
                         uv = None
                     else:
                         # Clamp slightly outside [0, 1] due to float precision
                         uv = np.clip(uv, 0.0, 1.0)
                         print(f"DEBUG Mesh {mesh_index}: Using normalized existing UVs.")

                else:
                     # Normalize valid UVs to [0,1] range for consistency
                     print(f"DEBUG Mesh {mesh_index}: Normalizing valid existing UVs to [0,1].")
                     uv = (uv - min_uv) / uv_range
                     uv = np.clip(uv, 0.0, 1.0) # Ensure they are strictly within [0,1]

    # If no valid UV mapping was found or derived from the mesh, compute using PCA.
    if uv is None:
        print(f"DEBUG Mesh {mesh_index}: No valid existing UVs found. Computing UV mapping using PCA.")
        uv = compute_pca_uv(mesh)
        if uv is None:
             print(f"ERROR Mesh {mesh_index}: Failed to compute PCA UVs.")
             # Cannot proceed with this mesh part
             return None

    # Final check on computed/processed UVs
    if uv is None or uv.shape[0] != mesh.vertices.shape[0]:
         print(f"ERROR Mesh {mesh_index}: Final UVs are invalid or mismatched (UV shape: {uv.shape if uv is not None else 'None'}, Vertices: {mesh.vertices.shape[0]})")
         return None

    print(f"--- Finished UV processing for mesh component {mesh_index} ---")
    return uv


def create_multi_chart_geometry_image(meshes, resolution=256):
    """Creates a multi-chart geometry image by packing individual mesh parameterizations."""
    if not meshes:
        print("ERROR: No mesh components provided to create_multi_chart_geometry_image.")
        # Return black image and default scale
        return np.zeros((resolution, resolution, 3), dtype=np.uint8), (0.0, 0.0)

    num_meshes = len(meshes)
    print(f"DEBUG: Creating multi-chart image for {num_meshes} mesh components.")

    # --- 1. Get or Compute UVs for each mesh ---
    mesh_data = [] # Store tuples of (mesh, uv)
    valid_meshes_indices = []
    for i, mesh in enumerate(meshes):
        uv = get_or_compute_uv_for_mesh(mesh, i)
        if uv is not None:
             # Ensure UVs are per-vertex
            if uv.shape[0] == mesh.vertices.shape[0]:
                mesh_data.append({'mesh': mesh, 'uv': uv, 'id': i})
                valid_meshes_indices.append(i)
            else:
                 print(f"ERROR: Mesh {i} UV shape {uv.shape} mismatch with vertex count {mesh.vertices.shape[0]}. Skipping this component.")
        else:
             print(f"WARNING: Skipping mesh component {i} due to failed UV generation.")

    if not mesh_data:
        print("ERROR: No valid mesh components with UVs could be processed.")
        return np.zeros((resolution, resolution, 3), dtype=np.uint8), (0.0, 0.0)

    num_valid_meshes = len(mesh_data)
    print(f"DEBUG: Proceeding with {num_valid_meshes} valid mesh components.")


    # --- 2. Plan Chart Packing (Simple Grid Layout) ---
    # Determine grid dimensions (nx * ny >= num_valid_meshes)
    nx = math.ceil(math.sqrt(num_valid_meshes))
    ny = math.ceil(num_valid_meshes / nx)
    print(f"DEBUG: Packing charts into a {nx} x {ny} grid.")

    cell_w = resolution / nx
    cell_h = resolution / ny

    packing_info = []
    for idx, data in enumerate(mesh_data):
        grid_x = idx % nx
        grid_y = idx // nx

        # Calculate offset and scale for this chart within the final image
        # Offset is the top-left corner of the cell in pixels
        offset_x = grid_x * cell_w
        offset_y = grid_y * cell_h
        # Scale determines how the [0,1] UV range maps to the cell dimensions
        # We might add a small margin inside each cell
        margin = 0.02 # 2% margin relative to cell size
        scale_x = cell_w * (1.0 - 2 * margin)
        scale_y = cell_h * (1.0 - 2 * margin)
        offset_x += cell_w * margin
        offset_y += cell_h * margin

        packing_info.append({
            'offset': np.array([offset_x, offset_y]),
            'scale': np.array([scale_x, scale_y]),
            'mesh_id': data['id'] # Original index
        })
        # Store packing info back with the mesh data for convenience
        data['packing'] = packing_info[-1]


    # --- 3. Rasterize Each Chart onto the Geometry Image ---
    geom_img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    pixel_count_per_mesh = np.zeros(num_valid_meshes, dtype=int)

    for chart_idx, data in enumerate(mesh_data):
        mesh = data['mesh']
        uv = data['uv'] # These are normalized [0,1] for this mesh part
        packing = data['packing']
        offset = packing['offset']
        scale = packing['scale']

        print(f"DEBUG: Rasterizing chart {chart_idx} (Mesh ID {packing['mesh_id']}) into cell @ ({offset[0]:.1f}, {offset[1]:.1f}) with scale ({scale[0]:.1f}, {scale[1]:.1f})")

        if not hasattr(mesh, 'faces') or mesh.faces.size == 0:
            print(f"DEBUG: Mesh ID {packing['mesh_id']} has no faces, skipping rasterization.")
            continue

        # Iterate through faces of the current mesh component
        for face_idx, face in enumerate(mesh.faces):
            try:
                # Get vertices and per-vertex UVs for this face
                verts = mesh.vertices[face]      # shape (3,3)
                uv_face = uv[face]              # shape (3,2), should be in [0,1] range

                # Transform face UVs according to packing info to get pixel coordinates
                # uv_pixels = uv_face * scale + offset
                # Ensure broadcasting works correctly: scale (2,) -> (1,2), offset (2,) -> (1,2)
                uv_pixels = uv_face * scale.reshape(1, 2) + offset.reshape(1, 2) # shape (3,2)

                # Compute the bounding box in pixel space for this face
                min_x = int(np.floor(np.min(uv_pixels[:, 0])))
                max_x = int(np.ceil(np.max(uv_pixels[:, 0])))
                min_y = int(np.floor(np.min(uv_pixels[:, 1])))
                max_y = int(np.ceil(np.max(uv_pixels[:, 1])))

                # Clip bounding box to image boundaries
                min_x = max(0, min_x)
                max_x = min(resolution - 1, max_x)
                min_y = max(0, min_y)
                max_y = min(resolution - 1, max_y)

                # Check if bounding box is valid
                if min_x > max_x or min_y > max_y:
                    continue # Skip if bbox is outside image or invalid


                # Loop over pixels in the bounding box
                for i in range(min_x, max_x + 1):
                    for j in range(min_y, max_y + 1):
                        # Pixel center coordinates
                        p = np.array([i + 0.5, j + 0.5])

                        # Check if pixel p is inside the triangle defined by uv_pixels
                        w = barycentric_weights(uv_pixels, p)

                        # If weights are valid (all >= 0), the pixel is inside the triangle
                        if np.all(w >= -1e-6): # Allow for slight numerical error
                            # Interpolate the 3D position using barycentric weights
                            # Ensure weights sum roughly to 1, normalize if needed (though barycentric should handle this)
                            # w_norm = w / w.sum() # Usually not needed if barycentric function is correct
                            pos = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]

                            # Assign the 3D position to the geometry image pixel
                            # Image uses [row, col] -> [y, x] indexing
                            geom_img[j, i, :] = pos
                            pixel_count_per_mesh[chart_idx] += 1

            except IndexError as e:
                 print(f"ERROR: IndexError during rasterization of chart {chart_idx}, face {face_idx}. Face indices: {face}. Vertices: {len(mesh.vertices)}, UVs: {len(uv)}. Error: {e}")
                 # This often happens if face indices are out of bounds for vertices or UVs
                 continue # Skip this face
            except Exception as e:
                 print(f"ERROR: Unexpected error during rasterization of chart {chart_idx}, face {face_idx}: {e}")
                 continue # Skip this face


    total_pixels_filled = pixel_count_per_mesh.sum()
    print(f"DEBUG: Rasterization complete. Total pixels filled: {total_pixels_filled}")
    if total_pixels_filled == 0:
         print("WARNING: No pixels were filled in the geometry image. Check UVs and rasterization logic.")

    # --- 4. Normalize and Scale the Geometry Image ---
    # Find min/max values across the *entire* image (only non-zero pixels if possible?)
    # Using only non-zero pixels avoids background affecting scale, but requires finding them.
    # Let's find bounds over the whole image first for simplicity, then refine if needed.

    min_val = np.min(geom_img) # Might be 0 if background is large
    max_val = np.max(geom_img)

    # Alternative: find min/max only from filled pixels if performance allows
    filled_pixels_mask = geom_img.any(axis=2) # True where any channel is non-zero
    if np.any(filled_pixels_mask):
        filled_values = geom_img[filled_pixels_mask]
        min_val_filled = np.min(filled_values)
        max_val_filled = np.max(filled_values)
        print(f"DEBUG Geometry Image Stats (Filled Pixels): Min={min_val_filled}, Max={max_val_filled}")
        # Use filled bounds if they seem more reasonable than overall bounds
        # (e.g., if overall min is 0 due to background but actual geometry is far from origin)
        # Heuristic: If min_val is near zero and min_val_filled is significantly different, use filled.
        if abs(min_val) < 1e-6 and abs(min_val_filled) > 1e-3:
            min_val = min_val_filled
            print("DEBUG: Using min_val from filled pixels.")
        if abs(max_val - max_val_filled) > 1e-3 : # Use max from filled if different
             max_val = max_val_filled
             print("DEBUG: Using max_val from filled pixels.")


    print("DEBUG Final Geometry Image Stats:")
    print("  Min value:", min_val)
    print("  Max value:", max_val)

    value_range = max_val - min_val
    if value_range < 1e-6: # Check for constant image or very small range
        print("WARNING: Geometry image values have zero or near-zero range. Scaling might be problematic.")
        # Avoid division by zero. Output black or grey image?
        # Let's output a zero image and return 0 scale params.
        geom_img_scaled = np.zeros_like(geom_img, dtype=np.uint8)
        scale_params = (0.0, 0.0) # Indicate problematic scaling
    else:
        # Normalize to [0, 1] and scale to [0, 255]
        geom_img_normalized = (geom_img - min_val) / value_range
        geom_img_scaled = (geom_img_normalized * 255.0).astype(np.uint8)
        scale_params = (min_val, max_val) # Store the original min/max for reconstruction

    return geom_img_scaled, scale_params

def reconstruct_obj_from_geometry_image(geom_img, scale_params, resolution=256):
    """
    Reconstructs a mesh from the geometry image.
    NOTE: This creates a dense grid mesh based on the image pixels.
    It does NOT reconstruct the original topology or seams between charts.
    It produces a surface approximation defined by the image.
    """
    min_val, max_val = scale_params
    value_range = max_val - min_val

    print("DEBUG Reconstruction: Image shape:", geom_img.shape, "Scale params:", scale_params)

    if value_range < 1e-6:
         print("WARNING: Reconstruction scale range is zero or near-zero. Cannot reconstruct meaningful geometry.")
         return np.empty((0, 3)), np.empty((0, 3)) # Return empty vertices and faces


    # Reverse scaling from 8-bit back to original float coordinates.
    geom_float = geom_img.astype(np.float32) / 255.0 * value_range + min_val

    # Create vertices from pixel coordinates
    # Only include vertices where the original image was non-black (or non-background)
    # Assuming background is black (0,0,0) AFTER scaling. Check geom_img values.
    # If scale_params were (0,0), geom_img is all 0. Handled above.
    # If geom_img has 0 values, these might be background or actual points at origin.
    # Let's create vertices for *all* pixels for simplicity, matching the original script.
    # A refinement could be to filter vertices based on a threshold or alpha mask if available.

    vertices = geom_float.reshape(-1, 3) # Shape: (resolution*resolution, 3)

    # Create faces based on grid connectivity
    faces = []
    for j in range(resolution - 1): # Row index (y)
        for i in range(resolution - 1): # Column index (x)
            # Vertex indices in the flattened array
            idx_tl = j * resolution + i       # Top-left
            idx_tr = idx_tl + 1               # Top-right
            idx_bl = (j + 1) * resolution + i # Bottom-left
            idx_br = idx_bl + 1               # Bottom-right

            # Create two triangles per grid cell
            # Check if vertices involved are valid? (e.g., not background if filtering)
            # For now, create all faces.
            faces.append([idx_tl, idx_bl, idx_tr]) # Triangle 1: TL, BL, TR
            faces.append([idx_tr, idx_bl, idx_br]) # Triangle 2: TR, BL, BR

    faces = np.array(faces)

    print("DEBUG: Reconstructed mesh (grid) with {} vertices and {} faces".format(len(vertices), len(faces)))
    # Optional: Create a trimesh object to clean up (remove degenerate faces, unused vertices)
    # mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh_recon.remove_degenerate_faces()
    # mesh_recon.remove_unreferenced_vertices()
    # vertices = mesh_recon.vertices
    # faces = mesh_recon.faces
    # print("DEBUG: Cleaned reconstructed mesh to {} vertices and {} faces".format(len(vertices), len(faces)))

    return vertices, faces

def write_obj(file_path, vertices, faces):
    """Writes vertices and faces to an OBJ file."""
    if vertices.size == 0 or faces.size == 0:
        print(f"WARNING: Attempting to write empty mesh to {file_path}. Skipping.")
        # Create an empty file or just return
        open(file_path, 'w').close()
        return

    print(f"DEBUG: Writing {len(vertices)} vertices and {len(faces)} faces to {file_path}")
    with open(file_path, 'w') as f:
        for v in vertices:
            # Format with reasonable precision, avoid scientific notation if possible
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))

        # Check face winding? Assume consistent from reconstruction grid.
        for face in faces:
            # OBJ file indices are 1-based.
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
    print(f"DEBUG: Finished writing {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a .obj file into a multi-chart geometry image and reconstruct a new .obj from the image."
    )
    parser.add_argument("input_obj", type=str, help="Input .obj file")
    parser.add_argument("output_geo", type=str, help="Output geometry image PNG file")
    parser.add_argument("output_obj", type=str, help="Output reconstructed .obj file")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the geometry image")
    args = parser.parse_args()

    print(f"--- Loading Input OBJ: {args.input_obj} ---")
    mesh_components = load_obj_parts(args.input_obj)

    if not mesh_components:
         print(f"ERROR: Could not load any valid mesh components from {args.input_obj}. Exiting.")
         # Create empty output files?
         Image.fromarray(np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8)).save(args.output_geo)
         write_obj(args.output_obj, np.empty((0,3)), np.empty((0,3)))
         return # Exit script

    print(f"\n--- Creating Geometry Image (Resolution: {args.resolution}x{args.resolution}) ---")
    geom_img, scale_params = create_multi_chart_geometry_image(mesh_components, args.resolution)

    print(f"\n--- Saving Geometry Image: {args.output_geo} ---")
    try:
        Image.fromarray(geom_img).save(args.output_geo)
        print("DEBUG: Geometry image saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save geometry image to {args.output_geo}: {e}")
        # Decide if script should continue or exit
        # return # Optional: exit if saving fails

    # --- Reconstruction ---
    # Check if scaling was valid before attempting reconstruction
    if scale_params == (0.0, 0.0) and np.all(geom_img == 0):
        print("WARNING: Geometry image appears empty or invalid (scale 0). Reconstruction will produce an empty OBJ.")
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
    else:
        print(f"\n--- Reconstructing OBJ from Geometry Image ---")
        # No need to reload image, we have it in memory (geom_img)
        # geom_img_recon = np.array(Image.open(args.output_geo)) # Reloading is redundant
        vertices, faces = reconstruct_obj_from_geometry_image(geom_img, scale_params, args.resolution)

    print(f"\n--- Saving Reconstructed OBJ: {args.output_obj} ---")
    write_obj(args.output_obj, vertices, faces)

    print("\n--- Processing Complete ---")


if __name__ == "__main__":
    main()