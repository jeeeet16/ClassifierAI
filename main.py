import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import _KerasLazyLoader
from keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize the images
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names for CIFAR-10
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Visualize the first 16 training images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Limit the training and testing data
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Building the model
# model = models.Sequential()

# Correctly specifying input_shape in the first Conv2D layer
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# Compiling the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Training the model
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# # Evaluating the model
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss = {loss}")
# print(f"Accuracy = {accuracy}")

# Saving the model in keras format
# model.save('image_classifier.keras')

# Loading the model
model = models.load_model('image_classifier.keras')

# Load and preprocess a test image
img = cv.imread('rsz_2car.jpg')  # Change this path to your image path
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
img = cv.resize(img, (32, 32))  # Resize to match input shape
img = np.array(img) / 255.0  # Normalize the image
img = img.reshape((1, 32, 32, 3))  # Reshape for prediction

# Make a prediction
prediction = model.predict(img)
index = np.argmax(prediction)

# Display the prediction
plt.imshow(img[0], cmap=plt.cm.binary)
plt.title(f"Prediction: {class_names[index]}")
plt.axis('off')
plt.show()
