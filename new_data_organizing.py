import os
import shutil
import random
from sklearn.model_selection import LeaveOneGroupOut
from collections import defaultdict
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np

def create_datasets(data_dir, output_dir, merge_categories=False,  max_samples_per_category=900):
    categories = ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Beard-Pulling', 'Eyebrow-Pulling']
    valid_groups = set()
    for root, dirs, files in os.walk(data_dir):
        valid_groups.update([d for d in dirs if not d.startswith('.')])

    group_to_index = {group: i for i, group in enumerate(sorted(valid_groups))}
    index_to_group = {i: group for group, i in group_to_index.items()}
    group_labels = list(sorted(valid_groups))

    logo = LeaveOneGroupOut()
    random.seed(42)

    category_files = {category: [] for category in categories}
    category_group_indices = {category: [] for category in categories}
    
    for category in categories:
        print(category)
        collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups, merge_map)

    min_samples = min(len(category_files[cat]) for cat in categories if cat != 'Non-BFRB')
    non_bfrb_samples = int(2 * min_samples) 

    for category in categories:
        if category == 'Non-BFRB':
            target_samples = non_bfrb_samples
        else:
            target_samples = min_samples

        balance_category_data(category, category_files, category_group_indices, group_to_index, target_samples, max_samples_per_category)
    valid_group_combinations = get_valid_group_combinations(group_labels, categories, category_group_indices, group_to_index)

    for fold, (test_group, val_groups) in enumerate(valid_group_combinations):
        available_categories = [cat for cat in categories if group_to_index[test_group] in category_group_indices[cat]]
        for category in available_categories:
            image_files = category_files[category]
            groups = category_group_indices[category]
            test_indices = [i for i, g in enumerate(groups) if g == group_to_index[test_group]]
            val_group = select_valid_group(val_groups, category, category_group_indices, group_to_index)
            if not val_group:
                continue
            val_indices = [i for i, g in enumerate(groups) if g == group_to_index[val_group]]
            train_indices = [i for i in range(len(groups)) if groups[i] not in [group_to_index[test_group], group_to_index[val_group]]]

            setup_directories(fold, category, test_indices, val_indices, train_indices, image_files, groups, group_to_index, output_dir, index_to_group, test_group, val_group, group_labels)

def balance_category_data(category, category_files, category_group_indices, group_to_index, target_samples, is_non_bfrb=False):
    files = category_files[category]
    group_indices = category_group_indices[category]
    group_count = defaultdict(int)

    for index in group_indices:
        group_count[index] += 1

    total_available_samples = sum(group_count.values())
    if is_non_bfrb:
        desired_samples_per_group = 2 * (target_samples // len(group_count))
    else:
        desired_samples_per_group = target_samples // len(group_count)

    new_files = []
    new_group_indices = []
    actual_group_count = defaultdict(int)

    combined_list = list(zip(files, group_indices))
    random.shuffle(combined_list)

    for file, group_index in combined_list:
        if actual_group_count[group_index] < desired_samples_per_group:
            new_files.append(file)
            new_group_indices.append(group_index)
            actual_group_count[group_index] += 1

    category_files[category] = new_files
    category_group_indices[category] = new_group_indices


def collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups, merge_map):
    for group in valid_groups:
        for cat in merge_map:
            if category == cat or category == merge_map[cat]:
                actual_category = merge_map.get(cat, cat)
                category_path = os.path.join(data_dir, cat, group)
                if os.path.isdir(category_path):
                    print(category_path)
                    for file in os.listdir(category_path):
                        if not file.startswith('cropped_') and file.endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(category_path, file)
                            category_files[actual_category].append(full_path)
                            category_group_indices[actual_category].append(group_to_index[group])
        if category not in merge_map.values():
            category_path = os.path.join(data_dir, category, group)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if not file.startswith('cropped_') and file.endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(category_path, file)
                        category_files[category].append(full_path)
                        category_group_indices[category].append(group_to_index[group])

def select_valid_group(groups, category, category_group_indices, group_to_index):
    valid_groups = [group for group in groups if group_to_index[group] in category_group_indices[category]]
    random.shuffle(valid_groups)
    while valid_groups:
        candidate = valid_groups.pop(0)
        if category_group_indices[category].count(group_to_index[candidate]) > 0:
            return candidate
    return None


def get_valid_group_combinations(group_labels, categories, category_group_indices, group_to_index):
    valid_combinations = []
    for test_group in group_labels:
        test_group_index = group_to_index[test_group]
        test_categories = [cat for cat in categories if test_group_index in category_group_indices[cat]]

        remaining_groups = [g for g in group_labels if g != test_group]
        valid_remaining_groups = []
        for g in remaining_groups:
            group_index = group_to_index[g]
            if any(group_index in category_group_indices[cat] for cat in test_categories):  # Changed from 'all' to 'any'
                valid_remaining_groups.append(g)

        random.shuffle(valid_remaining_groups)
        if valid_remaining_groups:
            valid_combinations.append((test_group, valid_remaining_groups))
    return valid_combinations



def setup_directories(fold, category, test_indices, val_indices, train_indices, image_files, groups, group_to_index, output_dir, index_to_group, test_group, val_group, group_labels):
    fold_dir_name = f"fold_{fold+1}"
    fold_output_dir = os.path.join(output_dir, fold_dir_name)
    if not os.path.exists(fold_output_dir):
        os.makedirs(fold_output_dir, exist_ok=True)

    for group_type, indices in [('train', train_indices), ('validation', val_indices), ('test', test_indices)]:
        category_dir = os.path.join(fold_output_dir, group_type, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        for idx in indices:
            src = image_files[idx]
            basename = os.path.basename(src)
            shutil.copy(src, os.path.join(category_dir, basename))

    print(f"Category: {category}, Fold {fold+1}: Test on {test_group}, Validate on {val_group}")

def detect_and_crop_face(image_path, width_margin=0.6, height_margin=0.6):
    image = Image.open(image_path)
    image_array = np.array(image)

    detector = MTCNN()
    results = detector.detect_faces(image_array)
    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height

        delta_width = width * width_margin
        delta_height = height * height_margin
        x1 = max(0, int(x1 - delta_width))
        y1 = max(0, int(y1 - delta_height))
        x2 = min(image_array.shape[1], int(x2 + delta_width))
        y2 = min(image_array.shape[0], int(y2 + delta_height))

        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    return image

def crop_and_save_images(data_dir):

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('cropped_'):
                full_path = os.path.join(root, file)
                try:
                    processed_image = detect_and_crop_face(full_path, width_margin=1.5, height_margin=1.5)
                    cropped_path = os.path.join(root, f"cropped_{file}")
                    processed_image.save(cropped_path)
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")

if __name__ == '__main__':
    # crop_and_save_images('./BFRB data/BFRB data')
    create_datasets('./BFRB data', './dataset_separated_old', merge_categories=False)
