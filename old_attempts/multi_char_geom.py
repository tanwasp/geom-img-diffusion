#!/usr/bin/env python3
"""
Multi–Chart Geometry Image Creation Script

This script demonstrates a simplified pipeline for constructing geometry images 
from a 3D mesh with UV mapping. The key steps include:
  1. Loading the mesh and its UV mapping.
  2. Splitting the mesh into connected components (initial charts).
  3. Detecting duplicated vertices with distinct UVs (non–injectivity issues).
  4. Detecting crease edges by evaluating differences in UV coordinates.
  5. Adjusting the UV mapping for an approximate equal–area projection.
  6. Sampling the UV domain to create a geometry image (where each pixel maps to a 3D point).

Usage:
  python multi_chart_geom.py --mesh path/to/mesh.obj --resolution 256 --output output_folder
"""

import argparse
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def load_mesh(mesh_path):
    """
    Load a 3D mesh and check for UV mapping.
    
    Parameters:
        mesh_path (str): Path to the mesh file.
    
    Returns:
        mesh (trimesh.Trimesh): Loaded mesh with UV mapping.
    
    Raises:
        ValueError: If the mesh does not contain a UV mapping.
    """
    mesh = trimesh.load(mesh_path, process=False)
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        raise ValueError("Mesh does not have UV mapping! Please ensure your mesh file contains UV coordinates.")
    return mesh

def get_connected_components(mesh):
    """
    Split the mesh into connected components.
    
    This serves as an initial separation into charts based on connectivity.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
    
    Returns:
        List[trimesh.Trimesh]: A list of meshes, each representing a connected component.
    """
    components = mesh.split(only_watertight=False)
    return components

def detect_duplicate_vertices(mesh, uv_tolerance=1e-5):
    """
    Detect vertices that are duplicates in 3D space but have distinct UV coordinates.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        uv_tolerance (float): Tolerance to compare UV coordinates.
    
    Returns:
        np.ndarray: Indices of vertices that are duplicated with different UVs.
    """
    vertices = mesh.vertices
    uv_coords = mesh.visual.uv
    duplicate_dict = {}
    for i, v in enumerate(vertices):
        # Use rounded vertex positions as keys to group near–duplicates
        key = tuple(np.round(v, decimals=5))
        duplicate_dict.setdefault(key, []).append(i)
    
    duplicate_indices = []
    for key, indices in duplicate_dict.items():
        if len(indices) > 1:
            # If the grouped vertices have differing UVs, flag them
            uvs = uv_coords[indices]
            if not np.allclose(uvs, uvs[0], atol=uv_tolerance):
                duplicate_indices.extend(indices)
    return np.unique(duplicate_indices)

def detect_creases(mesh, uv_threshold=0.1):
    """
    Detect crease edges by finding edges with a large difference in UV coordinates.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        uv_threshold (float): Threshold for the UV difference to consider an edge a crease.
    
    Returns:
        List[np.ndarray]: List of edge pairs (vertex indices) marked as creases.
    """
    crease_edges = []
    uv_coords = mesh.visual.uv
    for edge in mesh.edges_unique:
        diff = np.linalg.norm(uv_coords[edge[0]] - uv_coords[edge[1]])
        if diff > uv_threshold:
            crease_edges.append(edge)
    return crease_edges

def adjust_uv_equal_area(mesh, chart_face_indices):
    """
    Adjust the UV mapping for a given chart to approximate an equal-area projection.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        chart_face_indices (np.ndarray): Indices of faces belonging to the chart.
    
    Returns:
        mesh (trimesh.Trimesh): The mesh with adjusted UV mapping (for the specified faces).
    """
    faces = mesh.faces[chart_face_indices]
    vertices = mesh.vertices
    # Compute the 3D area of the chart
    areas = mesh.area_faces[chart_face_indices]
    total_area = np.sum(areas)
    
    # Compute the area in UV space for each face using a 2D triangle area formula
    uv_coords = mesh.visual.uv
    uv_faces = uv_coords[faces]
    def triangle_area(pts):
        # Using the shoelace formula for 2D triangles
        return 0.5 * np.abs(np.linalg.det(np.array([[pts[0,0], pts[0,1], 1],
                                                      [pts[1,0], pts[1,1], 1],
                                                      [pts[2,0], pts[2,1], 1]])))
    uv_areas = np.array([triangle_area(tri) for tri in uv_faces])
    total_uv_area = np.sum(uv_areas)
    
    # Compute a scaling factor that aligns the UV area with the 3D area
    scale = np.sqrt(total_area / total_uv_area) if total_uv_area > 0 else 1.0
    
    # Apply scaling to the UV coordinates for the vertices used in these faces
    # (This is a simplified version; a robust implementation might need to treat each chart separately.)
    face_vertex_indices = np.unique(faces.flatten())
    mesh.visual.uv[face_vertex_indices] = mesh.visual.uv[face_vertex_indices] * scale
    return mesh

def create_geometry_image(mesh, resolution=256):
    """
    Create a geometry image by sampling the mesh's 3D positions over a uniform grid in UV space.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        resolution (int): The resolution of the output geometry image.
    
    Returns:
        np.ndarray: A (resolution x resolution x 3) geometry image.
    """
    # Create a grid over the UV domain [0, 1] x [0, 1]
    grid_x, grid_y = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]
    
    # Get the UV coordinates and corresponding 3D vertex positions
    uv_coords = mesh.visual.uv
    vertices = mesh.vertices
    
    # Interpolate each coordinate (x, y, z) using griddata
    geom_image = np.zeros((resolution, resolution, 3))
    for i in range(3):
        geom_image[..., i] = griddata(uv_coords, vertices[:, i], (grid_x, grid_y), method='linear', fill_value=0)
    return geom_image

def main(args):
    # Load the mesh from file
    mesh = load_mesh(args.mesh)
    
    # Split the mesh into connected components (each serving as a preliminary chart)
    charts = get_connected_components(mesh)
    print(f"Found {len(charts)} connected component(s).")
    
    for i, chart in enumerate(charts):
        print(f"\nProcessing chart {i+1}")
        
        # Detect duplicate vertices with conflicting UVs
        dup_indices = detect_duplicate_vertices(chart)
        print(f"  Detected {len(dup_indices)} duplicated vertices with distinct UVs.")
        
        # Detect crease edges where the UV mapping changes abruptly
        crease_edges = detect_creases(chart)
        print(f"  Detected {len(crease_edges)} crease edge(s).")
        
        # (Optional) Further split the chart along crease edges if needed.
        # For simplicity, we continue with the chart as is.
        
        # Adjust the chart's UV mapping for an approximate equal–area projection
        face_indices = np.arange(len(chart.faces))
        chart = adjust_uv_equal_area(chart, face_indices)
        
        # Create the geometry image by sampling the 3D positions over UV space
        geom_img = create_geometry_image(chart, resolution=args.resolution)
        
        # Save the geometry image as a numpy array file and as a PNG for visualization
        out_npy = os.path.join(args.output, f"chart_{i+1}_geom.npy")
        np.save(out_npy, geom_img)
        print(f"  Saved geometry image (npy) to: {out_npy}")
        
        # Normalize image for visualization (scale to 0–1) and save as PNG
        norm_img = (geom_img - geom_img.min()) / (geom_img.max() - geom_img.min() + 1e-8)
        plt.figure()
        plt.imshow(norm_img)
        plt.title(f"Geometry Image for Chart {i+1}")
        plt.axis('off')
        out_png = os.path.join(args.output, f"chart_{i+1}_geom.png")
        plt.savefig(out_png, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization (png) to: {out_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi–Chart Geometry Image Creation")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the 3D mesh file (with UV mapping)")
    parser.add_argument("--resolution", type=int, default=256, help="Output geometry image resolution")
    parser.add_argument("--output", type=str, default="output", help="Directory to save output images")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    main(args)
