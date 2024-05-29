#**** Purpose: Fine-tuning pre-trained CNN for our application

# importing the required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from keras import models, layers, optimizers
from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Model



# ******************************************* Fine-tuning VGG16 model *****************************************************
# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for VGG16
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224), # matching the VGG input size
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for VGG-NET
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)


# Loading the VGGNet model
vgg16_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freezing the convolutional base layers
for layer in vgg16_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(224, 224, 3))
x = vgg16_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG16 with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg_imagenet_accuracy.png')

# plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG16 with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_vgg16_model.keras'):
    os.remove('trained_vgg16_model.keras')

model.save('trained_vgg16_model.keras')

# **********************************************************************************************************************


# ************************** Fine-tuning Xception model ****************************************************************

# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for Xception
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(299, 299), # matching the Xception input size
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for Xception
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)


# Loading the base xception model
xception_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# freezing the convolutional base layers
for layer in xception_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(299, 299, 3))
x = xception_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Xception with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('xception_imagenet_accuracy.png')

# plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Xception with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Xception_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_Xception_model.keras'):
    os.remove('trained_Xception_model.keras')

model.save('trained_Xception_model.keras')


# ************************** Fine-tuning VGG19 model ****************************************************************
# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for VGG19
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224), # matching the VGG input size
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for VGG19
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Loading the VGGNet model
vgg19_base = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freezing the convolutional base layers
for layer in vgg19_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(224, 224, 3))
x = vgg19_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG19 with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg19_imagenet_accuracy.png')

# plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG19 with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg19_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_vgg19_model.keras'):
    os.remove('trained_vgg19_model.keras')

model.save('trained_vgg19_model.keras')


# *********************************** Fine-tuning the ResNet50 model ***************************************************
# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for VGG19
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224), # matching the ResNet50 input size
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for VGG19
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Loading the ResNet50 model
resnet50_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freezing the convolutional base layers
for layer in resnet50_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(224, 224, 3))
x = resnet50_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet50 with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('ResNet50_imagenet_accuracy.png')

# plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNet50 with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('ResNet50_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_resnet50_model.keras'):
    os.remove('trained_resnet50_model.keras')

model.save('trained_resnet50_model.keras')


# ***************************** Fine-tuning the InceptionResNetV2 ******************************************************
# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for InceptionResnetV2
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(299, 299), # matching the InceptionResNetV2 input size
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for InceptionResNetV2
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

# Loading the InceptionResNetV2 model
inceptionResnetV2_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# freezing the convolutional base layers
for layer in inceptionResnetV2_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(299, 299, 3))
x = inceptionResnetV2_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('InceptionResNetV2 with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('inceptionResnetV2_imagenet_accuracy.png')

# plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('InceptionResNetV2 with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('inceptionResnetV2_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_inceptionResnetV2_model.keras'):
    os.remove('trained_inceptionResnetV2_model.keras')

model.save('trained_inceptionResnetV2_model.keras')


# **************************** Fine-tuning of EfficientNetV2S **********************************************************

# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for EfficientNetV2L
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for EfficientNetV2L
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

# Loading the InceptionResNetV2 model
efficientNetV2s_base = tf.keras.applications.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# freezing the convolutional base layers
for layer in efficientNetV2s_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(299, 299, 3))
x = efficientNetV2s_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('EfficientNetV2S with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('efficientNetV2S_imagenet_accuracy.png')

# plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('EfficientNetV2S with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('efficientNetV2S_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_efficientNetV2S_model.keras'):
    os.remove('trained_efficientNetV2S_model.keras')

model.save('trained_efficientNetV2S_model.keras')


# ******************************* Fine-tuning of NasNetLarge ***********************************************************
# creating an instance of the ImageDataGenerator class for training data only
train_datagen = ImageDataGenerator(
    rescale=1/255, # normalizing images

    # random rotations for data augmentation
    rotation_range=10,

    # zoom for data augmentation
    zoom_range = 0.2,

    # horizontal flip for data augmentation
    horizontal_flip=True
)

# creating an instance of the ImageDataGenerator class for validation data only
val_datagen = ImageDataGenerator(
    rescale=1/255
)

# creating an iterator for training dataset that resizes each image too so that it is suitable for EfficientNetV2L
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(331, 331),
    batch_size=16,
    class_mode='categorical'
)

# creating an iterator for validation dataset that resizes each image too so that it is suitable for EfficientNetV2L
validation_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(331, 331),
    batch_size=16,
    class_mode='categorical'
)

# Loading the InceptionResNetV2 model
nasNetLarge_base = tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))

# freezing the convolutional base layers
for layer in nasNetLarge_base.layers:
    layer.trainable = False

# adding some extra layers on top
input_tensor = Input(shape=(331, 331, 3))
x = nasNetLarge_base(input_tensor)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(4, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# fitting the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, validation_data = validation_generator)


# plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('NasNetLarge with imagenet Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('nasNetLarge_imagenet_accuracy.png')

# plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('NasNetLarge with imagenet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('nasNetLarge_imagenet_loss.png')

# Saving the model once trained.
# Check if the file exists and delete it
if os.path.exists('trained_efficientNetV2L_model.keras'):
    os.remove('trained_nasNetLarge_model.keras')

model.save('trained_nasNetLarge_model.keras')
