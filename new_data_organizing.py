import os
import shutil
from random import seed, sample

seed(42)

def create_directories(base_path, categories):
    for i in range(9):
        for subset in ['train', 'test', 'validation']:
            for category in categories:
                dir_path = os.path.join(base_path, f'fold_{i+1}', subset, category)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")

def distribute_files(source_path, base_path):
    categories = [cat for cat in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, cat))]

    create_directories(base_path, categories)

    for i, category in enumerate(categories):
        category_path = os.path.join(source_path, category)
        person_ids = sorted([d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))])
        for j, test_person in enumerate(person_ids):
            print(f"Processing fold for test person: {test_person} in category: {category}")
            # Setup paths for train, test, validation
            train_persons = [p for p in person_ids if p != test_person]

            # distribute test data
            test_source_folder = os.path.join(category_path, test_person)
            distribute_files_to_set(test_source_folder, os.path.join(base_path, f'fold_{j+1}', 'test', category), test_person)

            # get train data
            for train_person in train_persons:
                train_source_folder = os.path.join(category_path, train_person)
                distribute_files_to_set(train_source_folder, os.path.join(base_path, f'fold_{j+1}', 'train', category), train_person)

            # Create validation set by sampling from train set
            train_set_path = os.path.join(base_path, f'fold_{j+1}', 'train', category)
            all_files = [os.path.join(train_set_path, f) for f in os.listdir(train_set_path)
                         if os.path.isfile(os.path.join(train_set_path, f)) and
                         f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('cropped_')]
            if all_files:
                sample_size = min(len(all_files) // 6, len(all_files))
                validation_files = sample(all_files, sample_size) if sample_size > 0 else []
                for file_path in validation_files:
                    new_filename = f"{os.path.basename(file_path).split('.')[0]}_{os.path.basename(train_set_path)}.{'.'.join(os.path.basename(file_path).split('.')[1:])}"
                    shutil.move(file_path, os.path.join(base_path, f'fold_{j+1}', 'validation', category, new_filename))

def distribute_files_to_set(source_folder, dest_base_path, person_id):
    print(f"Copying files from {source_folder}")
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('cropped_'):
            new_filename = f"{person_id}_{file}"
            shutil.copy(file_path, os.path.join(dest_base_path, new_filename))

if __name__ == "__main__":
    source_path = os.path.join('.', 'BFRB data', 'BFRB data')
    base_path = os.path.join('.', 'processed_data')
    distribute_files(source_path, base_path)
