import os
import shutil
from random import seed, choice, sample
import time

seed(42)

def create_directories(base_path, categories):
    for i in range(9):  # Assuming there are 9 person_ids
        for subset in ['train', 'test', 'validation']:
            for category in categories:
                dir_path = os.path.join(base_path, f'fold_{i+1}', subset, category)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")

def distribute_files(source_path, base_path):
    cat_labels = ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Beard-Pulling', 'Eyebrow-Pulling']
    categories = [cat for cat in os.listdir(source_path) if (os.path.isdir(os.path.join(source_path, cat)) and cat in cat_labels)]
    global_person_ids = sorted({d for cat in categories for d in os.listdir(os.path.join(source_path, cat)) if os.path.isdir(os.path.join(source_path, cat, d))})

    create_directories(base_path, categories)

    for i, person_id in enumerate(global_person_ids):
        for category in categories:
            category_path = os.path.join(source_path, category)
            person_path = os.path.join(category_path, person_id)
            image_limit = 1500 if category == 'Non-BFRB' else 750
            print(f"Processing category: {category}, Fold: {i+1}")

            train_persons = [p for p in global_person_ids if p != person_id]
            valid_train_persons = [p for p in train_persons if os.path.exists(os.path.join(category_path, p)) and os.listdir(os.path.join(category_path, p))]
            if valid_train_persons:
                validation_person = choice(valid_train_persons)
                validation_person_path = os.path.join(category_path, validation_person)
                distribute_files_to_set(validation_person_path, os.path.join(base_path, f'fold_{i+1}', 'validation', category), validation_person)

            if os.path.exists(person_path) and os.listdir(person_path):
                distribute_files_to_set(person_path, os.path.join(base_path, f'fold_{i+1}', 'test', category), person_id)

            remaining_train_persons = [p for p in valid_train_persons if p != validation_person]
            person_image_dict = {}
            for train_person in remaining_train_persons:
                train_person_path = os.path.join(category_path, train_person)
                person_images = [os.path.join(train_person_path, f) for f in os.listdir(train_person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('cropped_')]
                person_image_dict[train_person] = person_images
                print(f"Person {train_person} has {len(person_images)} images in category {category}")

            all_images = [img for sublist in person_image_dict.values() for img in sublist]
            if len(all_images) > image_limit:
                selected_images = sample(all_images, image_limit)
            else:
                selected_images = all_images
            print(f"Selected {len(selected_images)} images for training in category {category}")

            for image_path in selected_images:
                shutil.copy(image_path, os.path.join(base_path, f'fold_{i+1}', 'train', category, unique_filename(image_path)))

def distribute_files_to_set(source_folder, dest_base_path, person_id):
    print(f"Copying files from {source_folder} to {dest_base_path}")
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('cropped_'):
            shutil.copy(file_path, os.path.join(dest_base_path, unique_filename(file_path, person_id)))

def unique_filename(file_path, person_id=None):
    # Generate a unique filename by adding a timestamp
    base_name = os.path.basename(file_path)
    name_part, extension = os.path.splitext(base_name)
    timestamp = int(time.time() * 1000)  # Millisecond timestamp
    if person_id:
        new_name = f"{person_id}_{name_part}_{timestamp}{extension}"
    else:
        new_name = f"{name_part}_{timestamp}{extension}"
    return new_name

if __name__ == "__main__":
    source_path = os.path.join('.', 'BFRB data')
    base_path = os.path.join('.', 'dataset_new')
    distribute_files(source_path, base_path)

