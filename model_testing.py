##**** Purpose: Testing the fitted models

# importing required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#**************************** Testting the Fine-tuned Xception model ***************************************************
# Loading the model
model = load_model('./trained_Xception_updated_model.keras')

## creating an iterator for test dataset that resizes each image too so that it is suitable for Xception
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset_separated/fold_7/test',
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using Xception')
plt.savefig('classification_report_xception.png', dpi=300, bbox_inches='tight')

# saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion Matrix using Xception')
plt.savefig('confusion_matrix_xception.png')

# ************************ Testing the fine-tuned NasNetLarge **********************************************************

# Loading the saved model
model = load_model('trained_nasNetLarge_model.keras')

# Creating an iterator for the test dataset that resizes each image so that it is suitable for VGG19 model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset_separated/fold_7/test',
    target_size=(331, 331),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# # saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using NasNetLarge')
plt.savefig('classification_report_nasNetLarge.png', dpi=300, bbox_inches='tight')

# # saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion matrix using NasNetLarge')
plt.savefig('confusion_matrix_nasNetLarge.png')

exit()


# ************************* Testing fine-tuned VGG-16 model*************************************************
# Loading the model
model = load_model('trained_vgg16_model.keras')

## creating an iterator for test dataset that resizes each image too so that it is suitable for VGG-NET
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using VGG16')
plt.savefig('classification_report_vgg16.png', dpi=300, bbox_inches='tight')


# saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion Matrix using VGG16')
plt.savefig('confusion_matrix_vgg16.png')


# **************************** Testing the Fine-tuned VGG19 model ******************************************************

# # Loading the saved model
model = load_model('trained_vgg19_model.keras')

# Creating an iterator for the test dataset that resizes each image so that it is suitable for VGG19 model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# # saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using VGG19')
plt.savefig('classification_report_vgg19.png', dpi=300, bbox_inches='tight')

# # saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion Matrix Using VGG19')
plt.savefig('confusion_matrix_vgg19.png')



# ********************************** Testing the Fine-tuned ResNet50 model**********************************************

# Loading the saved model
model = load_model('trained_resnet50_model.keras')

# Creating an iterator for the test dataset that resizes each image so that it is suitable for VGG19 model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# # saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using ResNet50')
plt.savefig('classification_report_resnet50.png', dpi=300, bbox_inches='tight')

# # saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion matrix using ResNet50')
plt.savefig('confusion_matrix_resnet50.png')


# ************************************** Testing the Fine-tuned InceptionResNetV2 model ********************************

# Loading the saved model
model = load_model('trained_inceptionResnetV2_model.keras')

# Creating an iterator for the test dataset that resizes each image so that it is suitable for VGG19 model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# # saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using InceptionResNetV2')
plt.savefig('classification_report_inceptionResnetV2.png', dpi=300, bbox_inches='tight')

# # saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion matrix using InceptionResNetV2')
plt.savefig('confusion_matrix_inceptionResNetV2.png')



# *************************** Testing the fine-tuned EfficientNetV2L ***************************************************

# Loading the saved model
model = load_model('trained_efficientNetV2S_model.keras')

# Creating an iterator for the test dataset that resizes each image so that it is suitable for VGG19 model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# making predictions using the model
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)

# checking the model performance
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# # saving the classification report
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center',
        fontsize=12, transform=ax.transAxes)
plt.axis('off')
plt.title('Classification report using EfficientNetV2S')
plt.savefig('classification_report_efficientNetV2S.png', dpi=300, bbox_inches='tight')

# # saving the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.title('Confusion matrix using EfficientNetV2S')
plt.savefig('confusion_matrix_efficientNetV2S.png')

