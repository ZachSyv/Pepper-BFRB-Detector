import tensorflow as tf
import os
import re
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    return image

def random_augment_image(image, seed):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
    image = tf.image.stateless_random_contrast(image, lower=0.8, upper=1.2, seed=seed)
    image = tf.image.stateless_random_saturation(image, lower=0.8, upper=1.2, seed=seed)
    image = tf.image.stateless_random_hue(image, max_delta=0.1, seed=seed)
    angle = tf.random.stateless_uniform([], minval=-15, maxval=15, seed=seed) * (3.14 / 180)
    image = tf.image.rot90(image, k=tf.cast(angle // (3.14 / 2), tf.int32))  # Rotate by multiples of 90 degrees
    crop_size = tf.random.stateless_uniform([], minval=0.9, maxval=1.0, seed=seed)
    image = tf.image.central_crop(image, central_fraction=crop_size)
    image = tf.image.resize(image, [299, 299])
    return image

def preprocess_train(sequence, seed):
    images = [load_and_preprocess_image(image_path) for image_path in sequence]
    images = [random_augment_image(image, seed) for image in images]
    return tf.stack(images)

def preprocess_val(sequence):
    images = [tf.image.resize(load_and_preprocess_image(image_path), [299, 299]) for image in sequence]
    return tf.stack(images)

def create_dataset(directory, batch_size, sequence_length, train=True):
    categories = sorted(os.listdir(directory))
    all_sequences = []
    all_labels = []

    pattern = re.compile(r"(\w+)_sequence_(\d+)\s\((\d+)\)\.jpg")

    for label, category in enumerate(categories):
        category_dir = os.path.join(directory, category)
        files = sorted(os.listdir(category_dir))
        sequence_dict = {}

        for file in files:
            match = pattern.match(file)
            if match:
                person_id, sequence_num, frame_num = match.groups()
                sequence_key = f"{person_id}_{sequence_num}"
                if sequence_key not in sequence_dict:
                    sequence_dict[sequence_key] = []
                sequence_dict[sequence_key].append((int(frame_num), os.path.join(category_dir, file)))

        for sequence_files in sequence_dict.values():
            sorted_files = sorted(sequence_files)  # Sort by frame number
            for i in range(0, len(sorted_files) - sequence_length + 1, sequence_length):  # Step by sequence_length
                sequence_group = [sorted_files[j][1] for j in range(i, i + sequence_length)]
                all_sequences.append(sequence_group)
                all_labels.append(label)

    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    dataset = tf.data.Dataset.from_tensor_slices((all_sequences, all_labels))

    def preprocess(sequence, label):
        seed = tf.random.uniform([2], maxval=1000, dtype=tf.int32)
        images = tf.py_function(func=lambda seq, sd: preprocess_train(seq, sd), inp=[sequence, seed], Tout=tf.float32)
        images.set_shape((sequence_length, 299, 299, 3))  # Set shape to avoid issues with dynamic shape
        return images, label

    dataset = dataset.map(preprocess) if train else dataset.map(lambda seq, lbl: (tf.numpy_function(func=preprocess_val, inp=[seq], Tout=tf.float32), lbl))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def build_model(sequence_length, num_classes, lstm_units=256, dropout_rate=0.5):
    base_model = Xception(weights='imagenet', include_top=False)

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(sequence_length, 299, 299, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dropout(dropout_rate)))  # Add Dropout after GlobalAveragePooling2D
    model.add(LSTM(lstm_units, return_sequences=True))  # Add return_sequences=True to stack another LSTM if needed
    model.add(Dropout(dropout_rate))  # Add Dropout after LSTM
    model.add(LSTM(lstm_units))  # Final LSTM layer
    model.add(Dropout(dropout_rate))  # Add Dropout before the final Dense layer
    model.add(Dense(num_classes, activation='softmax'))

    base_model.trainable = False  # Freeze the base model initially

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Train and evaluate the model for each fold
    num_folds = 9
    sequence_length = 4
    num_classes = 5  # Adjust this to the number of your categories
    batch_size = 16
    epochs = 5

    for fold_index in range(1, num_folds + 1):
        train_dir = f'./processed_data_sequence/fold_{fold_index}/train'
        val_dir = f'./processed_data_sequence/fold_{fold_index}/validation'
        
        train_dataset = create_dataset(train_dir, batch_size, sequence_length, train=True)
        val_dataset = create_dataset(val_dir, batch_size, sequence_length, train=False)

        # Build and compile the model
        model = build_model(sequence_length, num_classes)

        # Train the model
        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=epochs)

        # Save the model
        model.save(f'xception_lstm_model_fold_{fold_index}.keras')
