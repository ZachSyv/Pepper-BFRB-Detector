import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import glob

def setup_model(model_name, input_shape, num_categories):
    base_model = tf.keras.applications.__dict__[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    input_tensor = Input(shape=input_shape)
    x = base_model(input_tensor)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_categories, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    base_dir = 'dataset_new/'
    model_dir = 'models/'
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))
    folds = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(folds)
    model_configs = [
            {'model_name': 'VGG16', 'input_size': (224, 224, 3)},
            {'model_name': 'VGG19', 'input_size': (224, 224, 3)},
            {'model_name': 'Xception', 'input_size': (299, 299, 3)},
            {'model_name': 'ResNet50', 'input_size': (224, 224, 3)},
            {'model_name': 'InceptionResNetV2', 'input_size': (299, 299, 3)},
            {'model_name': 'EfficientNetV2S', 'input_size': (300, 300, 3)}, 
            {'model_name': 'NASNetLarge', 'input_size': (331, 331, 3)}
        ]
    
    fixed_categories = ['Beard-Pulling', 'Eyebrow-Pulling', 'Hair-Pulling', 'Nail-Biting', 'Non-BFRB']
    
    for model_path in model_files:
        modelfile_name = os.path.basename(model_path).replace('.keras', '')
        model_name = modelfile_name.split('_')[0]
        fold = modelfile_name.split('_')[-1]
        print(model_path)
        for config in model_configs:
            if config['model_name'] == model_name:
                input_size = config['input_size']
                break
    
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(base_dir, f'fold_{fold}', 'test'),
            target_size=input_size[:2],
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        model = setup_model(model_name, input_size, 5)
        model.load_weights(model_path)
        predictions = model.predict(test_generator, steps=len(test_generator))
        predicted_classes = np.argmax(predictions, axis=1)


        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
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

