import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model

# Directory and people setup
base_path = './BFRB data'
models_path = './models'
output_path = './output'
people = ['1', '2', '3', 'Angelica', 'Salah', 'Samir', 'Samuel', 'Shay', 'Zach']
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Class labels and indices
classes = ['Beard-Pulling', 'Eyebrow-Pulling', 'Hair-Pulling', 'Nail-Biting', 'Non-BFRB']

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


def process_video_frames(video_path, model, true_category, input_size):
    frame_files = sorted([p for p in os.listdir(video_path) if p.endswith(('.png', '.jpg', '.jpeg'))])
    chunk_size = 25
    num_chunks = len(frame_files) // chunk_size + (1 if len(frame_files) % chunk_size != 0 else 0)

    predictions = []

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(frame_files))
        chunk = frame_files[start_index:end_index]
        confident_predictions = {cls: 0 for cls in classes}

        for frame_file in chunk:
            frame_path = os.path.join(video_path, frame_file)
            frame = Image.open(frame_path).resize(input_size[:2])
            frame_array = np.array(frame).astype('float32') / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)
            
            prediction = model.predict(frame_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = classes[predicted_class_index]

            prediction_confidence = np.max(prediction)
            if prediction_confidence > 0.7:
                confident_predictions[predicted_class] += 1
                if confident_predictions[predicted_class] > 2:
                    predictions.append(predicted_class)
                    break
        else:
            predictions.append('Non-BFRB')

    return predictions


def generate_reports(predictions, true_labels, modelfile_name):
    unique_labels = np.unique(true_labels + predictions)
    target_names = [cls for cls in classes if cls in unique_labels]
    if target_names:  # Ensure target_names is not empty
        report = classification_report(true_labels, predictions, target_names=target_names, labels=unique_labels)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=ax.transAxes)
        plt.axis('off')
        plt.title(f'Classification Report for {modelfile_name}')
        plt.savefig(f'./output/classification_report_{modelfile_name}.png', dpi=300, bbox_inches='tight')
    else:
        print("No valid classes found for reporting.")

    cm = confusion_matrix(true_labels, predictions, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix for {modelfile_name}')
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes, rotation=45, ha="right")
    plt.savefig(f'./output/confusion_matrix_{modelfile_name}.png')

def process_models():
    model_configs = {
        'VGG16': (224, 224, 3),
        'VGG19': (224, 224, 3),
        'Xception': (299, 299, 3),
        'ResNet50': (224, 224, 3),
        'InceptionResNetV2': (299, 299, 3),
        'EfficientNetV2S': (300, 300, 3), 
        'NASNetLarge': (331, 331, 3)
    }
    results_by_architecture = {model: {'predictions': [], 'labels': []} for model in model_configs}

    for model_file in sorted(os.listdir(models_path)):
        model_path = os.path.join(models_path, model_file)
        model_name = model_file.split('_')[0]
        fold = model_file.split('_')[-1].split('.')[0]
        print(model_path)
        model = setup_model(model_name, input_size, 5)
        model.load_weights(model_path)
        input_size = model_configs[model_name]
        person_id = people[int(fold)-1]

        all_predictions = []
        all_true_labels = []

        cat_paths = [cat for cat in os.listdir(base_path) if cat in classes]
        for category in cat_paths:
            if person_id in os.listdir(os.path.join(base_path, category)):
                category_path = os.path.join(base_path, category, person_id)
                video_dirs = [v_dir for v_dir in os.listdir(category_path) if v_dir.lower().startswith('video')]
                for video_dir in video_dirs:
                    video_path = os.path.join(category_path, video_dir)
                    predictions = process_video_frames(video_path, model, category, input_size)
                    all_predictions.extend(predictions)
                    all_true_labels.extend([category] * len(predictions))

        generate_reports(all_predictions, all_true_labels, f'{model_name}_fold{fold}')
        results_by_architecture[model_name]['predictions'].extend(all_predictions)
        results_by_architecture[model_name]['labels'].extend(all_true_labels)

    # Generate aggregate reports for each architecture
    for model, data in results_by_architecture.items():
        generate_reports(data['predictions'], data['labels'], model)

if __name__ == "__main__":
    process_models()
