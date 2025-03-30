import os
import csv

# 1) Define your base directory and the car category folder.
BASE_DIR = "/Users/tanwasp/Documents/Research/ShapeNetCore"
car_dir = os.path.join(BASE_DIR, "02691156")

# 2) Get a sorted list of inner folders in the car folder.
inner_folders = sorted([
    folder for folder in os.listdir(car_dir)
    if os.path.isdir(os.path.join(car_dir, folder))
])

# 3) Define a function to parse an OBJ file and count vertices and faces.
def parse_obj_counts(obj_filepath):
    v_count = 0
    f_count = 0
    with open(obj_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                v_count += 1
            elif line.startswith('f '):
                f_count += 1
    return v_count, f_count

# 4) Process each inner folder and read the .obj file from its "models" subfolder.
results = []  # Will hold tuples: (folder_name, vertex_count, face_count)
for folder in inner_folders:
    models_dir = os.path.join(car_dir, folder, "models")
    obj_filepath = os.path.join(models_dir, "model_normalized.obj")
    
    if os.path.isfile(obj_filepath):
        vertices, faces = parse_obj_counts(obj_filepath)
        results.append((folder, vertices, faces))
    else:
        print(f"Warning: .obj file not found in {models_dir}")

# 5) Write the results to a CSV file.
csv_filename = "airplane_model_stats.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Folder", "Vertices Count", "Faces Count"])  # CSV header
    for folder, vertices, faces in results:
        writer.writerow([folder, vertices, faces])

print(f"CSV file '{csv_filename}' has been created with the model statistics.")
