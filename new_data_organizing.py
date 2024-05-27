import os
import shutil
import random
from sklearn.model_selection import LeaveOneGroupOut
from collections import defaultdict

def create_datasets(data_dir, output_dir, merge_categories=False):
    merge_map = {'Beard-Pulling': 'Facial Hair-Pulling', 'Eyebrow-Pulling': 'Facial Hair-Pulling'} if merge_categories else {}
    categories = ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Facial Hair-Pulling'] if merge_categories else ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Eyebrow-Pulling', 'Beard-Pulling']

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
        collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups, merge_map)

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

def collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups, merge_map):
    for group in valid_groups:
        for cat in merge_map:
            if category == cat or category == merge_map[cat]:
                actual_category = merge_map.get(cat, cat) 
                category_path = os.path.join(data_dir, cat, group)
                if os.path.isdir(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(category_path, file)
                            category_files[actual_category].append(full_path)
                            category_group_indices[actual_category].append(group_to_index[group])
        if category not in merge_map.values():
            category_path = os.path.join(data_dir, category, group)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(category_path, file)
                        category_files[category].append(full_path)
                        category_group_indices[category].append(group_to_index[group])

def select_valid_group(groups, category, category_group_indices, group_to_index):
    valid_groups = [group for group in groups if group_to_index[group] in category_group_indices[category]]
    random.shuffle(valid_groups)
    return valid_groups[0] if valid_groups else None

def get_valid_group_combinations(group_labels, categories, category_group_indices, group_to_index):
    valid_combinations = []
    for test_group in group_labels:
        test_group_index = group_to_index[test_group]
        test_categories = [cat for cat in categories if test_group_index in category_group_indices[cat]]

        remaining_groups = [g for g in group_labels if g != test_group]
        valid_remaining_groups = []
        for g in remaining_groups:
            group_index = group_to_index[g]
            if all(group_index in category_group_indices[cat] for cat in test_categories):
                valid_remaining_groups.append(g)

        random.shuffle(valid_remaining_groups)
        if valid_remaining_groups:
            valid_combinations.append((test_group, valid_remaining_groups))
    return valid_combinations


def setup_directories(fold, category, test_indices, val_indices, train_indices, image_files, groups, group_to_index, output_dir, index_to_group, test_group, val_group, group_labels):
    fold_dir_name = f"fold_{fold+1}_test_{test_group}_val_{val_group}"
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

def consolidate_and_rename_folds(src_base_dir, dst_base_dir):
    folds = defaultdict(list)

    for folder_name in os.listdir(src_base_dir):
        parts = folder_name.split('_')
        fold_prefix = parts[0] + '_' + parts[1] 
        folds[fold_prefix].append(os.path.join(src_base_dir, folder_name))

    sorted_fold_prefixes = sorted(folds.keys(), key=lambda x: int(x.split('_')[1]))

    for new_index, fold_prefix in enumerate(sorted_fold_prefixes, start=1):
        fold_dst_dir = os.path.join(dst_base_dir, f"fold_{new_index}")
        os.makedirs(fold_dst_dir, exist_ok=True)

        for dtype in ['train', 'validation', 'test']:
            dtype_dst_dir = os.path.join(fold_dst_dir, dtype)
            os.makedirs(dtype_dst_dir, exist_ok=True)

            for directory in folds[fold_prefix]:
                dtype_src_dir = os.path.join(directory, dtype)
                if os.path.exists(dtype_src_dir):
                    for category in os.listdir(dtype_src_dir):
                        category_dst_dir = os.path.join(dtype_dst_dir, category)
                        os.makedirs(category_dst_dir, exist_ok=True)

                        for file in os.listdir(os.path.join(dtype_src_dir, category)):
                            src_file = os.path.join(dtype_src_dir, category, file)
                            dst_file = os.path.join(category_dst_dir, file)
                            shutil.copy(src_file, dst_file)

if __name__ == '__main__':
    create_datasets('./BFRB data', './dataset_merged', merge_categories=True)
    consolidate_and_rename_folds('./dataset_merged', './merged')