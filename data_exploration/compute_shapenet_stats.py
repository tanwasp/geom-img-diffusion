import os
import csv
import statistics
import json

# 1) Define your base directory where the unzipped categories are stored.
BASE_DIR = "/Users/tanwasp/Documents/Research/ShapeNetCore"

# 2) Define your category map (WordNet offset -> readable category name).
category_map = {
    '02691156': 'airplane',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench',
    '02843684': 'birdhouse',
    '02871439': 'bookshelf',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02942699': 'camera',
    '02946921': 'can',
    '02954340': 'cap',
    '02958343': 'car',
    '02992529': 'cellular_telephone',
    '03001627': 'chair',
    '03046257': 'clock',
    '03085013': 'computer_keyboard',
    '03207941': 'dishwasher',
    '03211117': 'display',
    '03261776': 'earphone',
    '03325088': 'faucet',
    '03337140': 'file',
    '03467517': 'guitar',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'loudspeaker',
    '03710193': 'mailbox',
    '03759954': 'microphone',
    '03761084': 'microwave',
    '03790512': 'motorcycle',
    '03797390': 'mug',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '03991062': 'pot',
    '04004475': 'printer',
    '04074963': 'remote_control',
    '04090263': 'rifle',
    '04099429': 'rocket',
    '04225987': 'skateboard',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04379243': 'table',
    '04401088': 'telephone',
    '04460130': 'tower',
    '04468005': 'train',
    '02747177': 'trashcan',
    '04530566': 'vessel',
    '04554684': 'washer'
}

# 3) Function to parse an OBJ file and return (vertex_count, face_count).
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

# 4) Traverse each category directory, find all model_normalized.obj, parse them.
results = {}  # Will store category -> stats

for cat_id, cat_name in category_map.items():
    folder_name = f"{cat_id}"
    cat_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.isdir(cat_path):
        print(f"Warning: Directory for category {cat_name} ({cat_id}) not found at {cat_path}")
        continue
    
    vertices_list = []
    faces_list = []
    
    # Walk through all subdirectories looking for 'model_normalized.obj'
    for root, dirs, files in os.walk(cat_path):
        if 'model_normalized.obj' in files:
            obj_filepath = os.path.join(root, 'model_normalized.obj')
            v_count, f_count = parse_obj_counts(obj_filepath)
            vertices_list.append(v_count)
            faces_list.append(f_count)
    
    if len(vertices_list) == 0:
        print(f"No models found for category {cat_name} ({cat_id})")
        continue
    
    # Compute stats for the category
    min_vertices = min(vertices_list)
    max_vertices = max(vertices_list)
    avg_vertices = statistics.mean(vertices_list)
    
    min_faces = min(faces_list)
    max_faces = max(faces_list)
    avg_faces = statistics.mean(faces_list)
    
    results[cat_name] = {
        "num_models": len(vertices_list),
        "min_vertices": min_vertices,
        "max_vertices": max_vertices,
        "avg_vertices": avg_vertices,
        "min_faces": min_faces,
        "max_faces": max_faces,
        "avg_faces": avg_faces
    }

# 5) Write results to a CSV file.
csv_filename = "shapenet_stats_new.csv"
csv_fields = [
    "category", 
    "num_models", 
    "min_vertices", 
    "max_vertices", 
    "avg_vertices", 
    "min_faces", 
    "max_faces", 
    "avg_faces"
]

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_fields)  # Header row
    
    for cat_name, stats in results.items():
        writer.writerow([
            cat_name,
            stats["num_models"],
            stats["min_vertices"],
            stats["max_vertices"],
            f"{stats['avg_vertices']:.2f}",  # format average to 2 decimals if desired
            stats["min_faces"],
            stats["max_faces"],
            f"{stats['avg_faces']:.2f}"
        ])

print(f"CSV file '{csv_filename}' has been created with the ShapeNet statistics.")
