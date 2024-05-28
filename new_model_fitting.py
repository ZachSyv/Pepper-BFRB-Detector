import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from collections import defaultdict
import json

def setup_model(model_name, input_shape, num_categories):
    base_model = tf.keras.applications.__dict__[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    input_tensor = Input(shape=input_shape)
    x = base_model(input_tensor)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Added L2 regularization
    x = Dropout(0.50)(x) 
    output_tensor = Dense(num_categories, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_data_generators(train_dir, val_dir, image_size, batch_size=16):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2], 
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_generator, validation_generator

def get_fold_directories(base_dir):
    folds = defaultdict(list)
    for directory in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, directory)):
            fold = directory
            folds[fold].append(directory)
    return folds

def train_and_save_model(model, train_generator, validation_generator, epochs, fold_id, model_name, base_model):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    model_save_path = f'models/{model_name}_{base_dir[2:]}_{fold_id}.keras'
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    with open(f'history/{model_name}_{base_dir[2:]}_history_{fold_id}.json', 'w') as f:
        json.dump(history.history, f)

def train_across_folds(base_dir, model_config, epochs):
    folds = get_fold_directories(base_dir)
    for fold, directories in folds.items():
        print(f"Processing {model_config['model_name']} for Fold: {fold}")
        if(fold != 'fold_5'):
          for sub_directory in directories:
              train_dir = os.path.join(base_dir, sub_directory, 'train')
              val_dir = os.path.join(base_dir, sub_directory, 'validation')
              train_generator, validation_generator = get_data_generators(train_dir, val_dir, model_config['input_size'][:2])
              model = setup_model(model_config['model_name'], model_config['input_size'], len(os.listdir(train_dir)))
              train_and_save_model(model, train_generator, validation_generator, epochs, fold, model_config['model_name'], base_dir)


if __name__ == '__main__':    
    base_dirs = ['./separated']
    for base_dir in base_dirs:
        model_configs = [
            # {'model_name': 'VGG16', 'input_size': (224, 224, 3)},
            {'model_name': 'VGG19', 'input_size': (224, 224, 3)},
            {'model_name': 'Xception', 'input_size': (299, 299, 3)},
            {'model_name': 'ResNet50', 'input_size': (224, 224, 3)},
            {'model_name': 'InceptionResNetV2', 'input_size': (299, 299, 3)},
            {'model_name': 'EfficientNetV2S', 'input_size': (300, 300, 3)}, 
            {'model_name': 'NASNetLarge', 'input_size': (331, 331, 3)}
        ]
        epochs = 15

        for model_config in model_configs:
            train_across_folds(base_dir, model_config, epochs)

