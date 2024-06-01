import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os
import glob
import re

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])  # Assuming all models use 299x299 input size
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    return image

def random_augment_image(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.1)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_crop(image, size=[int(299 * 0.8), int(299 * 0.8), 3])
        image = tf.image.resize(image, [299, 299])
    return image

def preprocess_train(sequence):
    images = [random_augment_image(load_and_preprocess_image(image_path)) for image_path in sequence]
    images = [tf.image.resize(image, [299, 299]) for image in images]
    return tf.stack(images)

def preprocess_val(sequence):
    images = [tf.image.resize(load_and_preprocess_image(image_path), [299, 299]) for image_path in sequence]
    return tf.stack(images)

def create_dataset(directory, batch_size, sequence_length, train=True):
    categories = sorted(os.listdir(directory))
    all_sequences = []
    all_labels = []

    for label, category in enumerate(categories):
        category_dir = os.path.join(directory, category)
        files = sorted(os.listdir(category_dir))
        sequence_dict = {}

        for file in files:
            match = re.match(r"(\w+)_sequence_(\d+)\s\((\d+)\)\.jpg", file)
            if match:
                person_id, sequence_num, frame_num = match.groups()
                if sequence_num not in sequence_dict:
                    sequence_dict[sequence_num] = []
                sequence_dict[sequence_num].append((int(frame_num), os.path.join(category_dir, file)))

        for sequence_files in sequence_dict.values():
            sorted_files = sorted(sequence_files)
            for i in range(len(sorted_files) - sequence_length + 1):
                sequence_group = [sorted_files[j][1] for j in range(i, i + sequence_length)]
                all_sequences.append(sequence_group)
                all_labels.append(label)

    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    dataset = tf.data.Dataset.from_tensor_slices((all_sequences, all_labels))

    def preprocess(sequence, label):
        images = tf.numpy_function(func=preprocess_train if train else preprocess_val, inp=[sequence], Tout=tf.float32)
        images.set_shape((sequence_length, 299, 299, 3))  # Set shape to avoid issues with dynamic shape
        return images, label

    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, categories

if __name__ == '__main__':
    base_dir = './processed_data_sequence/'
    model_dir = 'models/'
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))

    folds = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(folds)

    sequence_length = 4
    fixed_categories = ['Hair-Pulling', 'Nail-Biting', 'Non-BFRB', 'Beard-Pulling', 'Eyebrow-Pulling']

    for model_path in model_files:
        modelfile_name = os.path.basename(model_path).replace('.keras', '')
        model_name = modelfile_name.split('_')[0]
        fold = modelfile_name.split('_')[-1]

        test_dir = os.path.join(base_dir, f'fold_{fold}', 'test')
        test_dataset, class_labels = create_dataset(test_dir, batch_size=1, sequence_length=sequence_length, train=False)

        model = load_model(model_path)
        predictions = model.predict(test_dataset, steps=len(test_dataset))
        predicted_classes = np.argmax(predictions, axis=1)

        true_classes = []
        for _, label in test_dataset:
            true_classes.append(label.numpy()[0])

        true_classes = np.array(true_classes)

        print("True classes:", true_classes)
        print("Predicted classes:", predicted_classes)
        print("Class labels:", class_labels)
        
        for category in fixed_categories:
            if category not in class_labels:
                class_labels.append(category)
        
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        
        print("Label to index mapping:", label_to_index)
        
        true_classes_mapped = [label_to_index[class_labels[label]] for label in true_classes]
        predicted_classes_mapped = [label_to_index[class_labels[label]] for label in predicted_classes]
        
        print("Mapped true classes:", true_classes_mapped)
        print("Mapped predicted classes:", predicted_classes_mapped)
        
        report = classification_report(true_classes_mapped, predicted_classes_mapped, target_names=class_labels, labels=range(len(class_labels)))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=ax.transAxes)
        plt.axis('off')
        plt.title(f'Classification Report for {modelfile_name}')
        plt.savefig(f'classification_report_{modelfile_name}.png', dpi=300, bbox_inches='tight')

        cm = confusion_matrix(true_classes_mapped, predicted_classes_mapped, labels=range(len(class_labels)))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title(f'Confusion Matrix for {modelfile_name}')
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels, rotation=45, ha="right")
        plt.savefig(f'confusion_matrix_{modelfile_name}.png')
