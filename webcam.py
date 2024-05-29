import sys
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class WebcamGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model_config = self.load_model_config()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.capture_button = QPushButton("Capture and Predict", self)
        self.capture_button.clicked.connect(self.capture_and_predict)
        self.layout.addWidget(self.capture_button)

        self.model_selector = QComboBox(self)
        self.load_model_paths()
        self.model_selector.currentIndexChanged.connect(self.model_selected)
        self.layout.addWidget(self.model_selector)

        self.model = load_model(self.model_selector.currentText())
        self.current_model_name = self.extract_model_name(self.model_selector.currentText())

        self.cap = cv2.VideoCapture(0)

        self.setWindowTitle('Behavior Prediction')
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def load_model_paths(self):
        models_dir = './models'
        for model_name in sorted(os.listdir(models_dir)):
            self.model_selector.addItem(os.path.join(models_dir, model_name))

    def model_selected(self, index):
        if self.model_selector.currentText():
            self.model = load_model(self.model_selector.currentText())
            self.current_model_name = self.extract_model_name(self.model_selector.currentText())

    def extract_model_name(self, model_path):
        return os.path.basename(model_path).split('_')[0]  # Assumes format "ModelName.h5"

    def load_model_config(self):
        return [
            {'model_name': 'VGG16', 'input_size': (224, 224, 3)},
            {'model_name': 'VGG19', 'input_size': (224, 224, 3)},
            {'model_name': 'Xception', 'input_size': (299, 299, 3)},
            {'model_name': 'ResNet50', 'input_size': (224, 224, 3)},
            {'model_name': 'InceptionResNetV2', 'input_size': (299, 299, 3)},
            {'model_name': 'EfficientNetV2S', 'input_size': (300, 300, 3)}, 
            {'model_name': 'NASNetLarge', 'input_size': (331, 331, 3)}
        ]

    def capture_and_predict(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.predict_behavior(rgb_image)
            self.display_image(rgb_image)
            self.setWindowTitle(f'Behavior Prediction - {predictions}')
        else:
            self.image_label.setText("Failed to capture image")

    def predict_behavior(self, image):
        # Get the correct input size for the current model
        input_size = next((config['input_size'] for config in self.model_config if config['model_name'] == self.current_model_name), (224, 224, 3))
        image = cv2.resize(image, input_size[:2])
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        return np.argmax(predictions)

    def display_image(self, image):
        qformat = QImage.Format.Format_RGB888
        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(out_image))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WebcamGUI()
    sys.exit(app.exec())
