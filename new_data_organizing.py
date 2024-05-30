import os
import shutil
import random
from sklearn.model_selection import LeaveOneGroupOut
from collections import defaultdict

def create_datasets(data_dir, output_dir, max_samples_per_category=400):
    categories = ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Beard-Pulling', 'Eyebrow-Pulling']
    valid_groups = set()

    for root, dirs, files in os.walk(data_dir):
        valid_groups.update([d for d in dirs if not d.startswith('.')])

    group_to_index = {group: i for i, group in enumerate(sorted(valid_groups))}
    category_files = {category: [] for category in categories}
    category_group_indices = {category: [] for category in categories}
    
    for category in categories:
        collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups)

    min_samples = min(len(category_files[cat]) for cat in categories if cat != 'Non-BFRB')
    non_bfrb_samples = int(2 * min_samples)

    for category in categories:
        target_samples = non_bfrb_samples if category == 'Non-BFRB' else min_samples
        balance_category_data(category, category_files, category_group_indices, group_to_index, target_samples, max_samples_per_category)

    valid_group_combinations = get_valid_group_combinations(categories, category_group_indices, group_to_index)

    for fold, (test_group, remaining_groups) in enumerate(valid_group_combinations):
        setup_fold(fold, test_group, remaining_groups, category_files, category_group_indices, group_to_index, output_dir, categories)

def collect_files(data_dir, category, category_files, category_group_indices, group_to_index, valid_groups):
    for group in valid_groups:
        category_path = os.path.join(data_dir, category, group)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('cropped_'):
                    full_path = os.path.join(category_path, file)
                    category_files[category].append(full_path)
                    category_group_indices[category].append(group_to_index[group])

def balance_category_data(category, category_files, category_group_indices, group_to_index, target_samples, max_samples_per_category):
    files = category_files[category]
    group_indices = category_group_indices[category]
    group_count = defaultdict(int)

    for index in group_indices:
        group_count[index] += 1

    desired_samples_per_group = target_samples // len(group_count)
    new_files, new_group_indices = [], []
    combined_list = list(zip(files, group_indices))
    random.shuffle(combined_list)

    for file, group_index in combined_list:
        if len(new_files) < max_samples_per_category and group_count[group_index] > 0:
            new_files.append(file)
            new_group_indices.append(group_index)
            group_count[group_index] -= 1

    category_files[category] = new_files
    category_group_indices[category] = new_group_indices

def get_valid_group_combinations(categories, category_group_indices, group_to_index):
    valid_combinations = []
    for group, index in group_to_index.items():
        other_groups = [g for g, idx in group_to_index.items() if idx != index]
        if any(index in category_group_indices[cat] for cat in categories):
            valid_combinations.append((group, other_groups))
    return valid_combinations

def setup_fold(fold, test_group, remaining_groups, category_files, category_group_indices, group_to_index, output_dir, categories):
    fold_dir_name = f"fold_{fold+1}"
    fold_output_dir = os.path.join(output_dir, fold_dir_name)
    os.makedirs(fold_output_dir, exist_ok=True)

    test_indices, train_indices = [], []
    for category in categories:
        image_files = category_files[category]
        groups = category_group_indices[category]
        category_test_indices = [i for i, g in enumerate(groups) if g == group_to_index[test_group]]
        category_train_indices = [i for i in range(len(groups)) if groups[i] != group_to_index[test_group]]

        distribute_files(fold_output_dir, category, image_files, category_test_indices, category_train_indices)

def distribute_files(fold_output_dir, category, image_files, test_indices, train_indices):
    random.shuffle(train_indices)
    num_val_samples = len(train_indices) // 6
    val_indices = train_indices[:num_val_samples]
    new_train_indices = train_indices[num_val_samples:]

    train_dir = os.path.join(fold_output_dir, 'train', category)
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(fold_output_dir, 'validation', category)
    os.makedirs(val_dir, exist_ok=True)

    if test_indices:
        test_dir = os.path.join(fold_output_dir, 'test', category)
        os.makedirs(test_dir, exist_ok=True)

    for idx in new_train_indices:
        src = image_files[idx]
        shutil.copy(src, os.path.join(train_dir, os.path.basename(src)))

    for idx in val_indices:
        src = image_files[idx]
        shutil.copy(src, os.path.join(val_dir, os.path.basename(src)))

    for idx in test_indices:
        src = image_files[idx]
        if 'test_dir' in locals():  # Check if the test_dir was created
            shutil.copy(src, os.path.join(test_dir, os.path.basename(src)))


if __name__ == '__main__':
    create_datasets('./BFRB data/BFRB data', './dataset_separated_new', max_samples_per_category=400)
