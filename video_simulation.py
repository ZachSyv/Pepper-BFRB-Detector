import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Directory structure and people array
base_path = './BFRB data'
models_path = './models'
output_path = './output'
people = ['1', '2', '3', 'Angelica', 'Salah', 'Samir', 'Samuel', 'Shay', 'Zach']
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Class labels and indices
class_indices = {'Beard-Pulling': 0, 'Eyebrow-Pulling': 1, 'Hair-Pulling': 2, 'Nail-Biting': 3, 'Non-BFRB': 4}
classes = ['Beard-Pulling', 'Eyebrow-Pulling', 'Hair-Pulling', 'Nail-Biting', 'Non-BFRB']

def process_video_frames(video_path, model, true_category, input_size):
    """Process frames in chunks and monitor behavior changes, returning a prediction for each chunk."""
    frame_files = sorted(os.listdir(video_path))
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
                    predictions.append(predicted_class if predicted_class == true_category else 'Incorrect detection')
                    break
        else:
            # If no confident detection over threshold is found in the chunk
            predictions.append('Non-BFRB' if true_category == 'Non-BFRB' else 'Incorrect detection')

    return predictions

def test_model(model, model_name, person_id, input_size):
    """Test the model and generate reports for predictions."""
    all_predictions = []
    all_true_labels = []

    # Adjusted path to correctly locate the video frames within each category
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category, "Video frames")
        for video_dir in os.listdir(category_path):
            if video_dir.startswith('Video'):
                video_path = os.path.join(category_path, video_dir)
                predictions = process_video_frames(video_path, model, category, input_size)
                all_predictions.extend(predictions)
                all_true_labels.extend([category] * len(predictions))

    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    return accuracy, all_predictions, all_true_labels

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
    results_by_architecture = {model: [] for model in model_configs}

    for model_file in sorted(os.listdir(models_path)):
        model_path = os.path.join(models_path, model_file)
        model_name = model_file.split('_')[0]
        model = load_model(model_path)
        input_size = model_configs[model_name]

        for person_id in people:
            accuracy, predictions, true_labels = test_model(model, model_name, person_id, input_size)
            results_by_architecture[model_name].append((accuracy, predictions, true_labels))

    for architecture, results in results_by_architecture.items():
        fold_accuracies = [result[0] for result in results]
        mean_accuracy = np.mean(fold_accuracies)
        
        # Concatenate all predictions and true labels across folds for detailed reports
        all_predictions = np.concatenate([result[1] for result in results])
        all_true_labels = np.concatenate([result[2] for result in results])

        # Generate overall reports
        report = classification_report(all_true_labels, all_predictions, target_names=classes)
        cm = confusion_matrix(all_true_labels, all_predictions, labels=range(len(classes)))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        # Save and display the report and confusion matrix
        print(f"Mean accuracy for {architecture}: {mean_accuracy}")
        print(report)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_display.plot(ax=ax, cmap='Blues')
        plt.title(f'Confusion Matrix for {architecture}')
        plt.savefig(os.path.join(output_path, f'confusion_matrix_{architecture}.png'))
        plt.close()

if __name__ == "__main__":
    process_models()
