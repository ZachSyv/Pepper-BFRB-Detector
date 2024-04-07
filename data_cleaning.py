#*****PURPOSE*****
# The purpose of this file is to get the dataset in required format so that we can use keras' ImageDataGenerator to perform data augmentation


# loading the required packages
import os
import shutil
from sklearn.model_selection import train_test_split

# Creating train, test, and validation directories
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)
os.makedirs('dataset/validation', exist_ok=True)

# Defining the categories
categories = ['Hair-Pulling', 'Beard-Pulling', 'Nail-Biting', 'Non-BFRB']

# Splitting the data into train, test, and validation sets.
for category in categories:

    # creating a directory for each category within train, test, and validation directories
    os.makedirs(os.path.join('dataset/train', category), exist_ok=True)
    os.makedirs(os.path.join('dataset/test', category), exist_ok=True)
    os.makedirs(os.path.join('dataset/validation', category), exist_ok=True)

    # getting a list of all files in dataset/images/[category]
    files = os.listdir(os.path.join('dataset/images', category))

    # splitting the files into train, test, and validation sets
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42) # since 0.25 x 0.8 = 0.2

    # copying the files in respective directories
    for file in train_files:
        shutil.copy(os.path.join('dataset/images', category, file), os.path.join('dataset/train', category, file))
    for file in test_files:
        shutil.copy(os.path.join('dataset/images', category, file), os.path.join('dataset/test', category, file))
    for file in val_files:
        shutil.copy(os.path.join('dataset/images', category, file), os.path.join('dataset/validation', category, file))