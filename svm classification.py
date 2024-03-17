#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')


# In[23]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(directory):
    images = []
    labels = []
    for label, category in enumerate(['cat', 'dog']):
        path = os.path.join(directory, category)
        for image_file in os.listdir(path):
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_file), target_size=(150, 150))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

train_dir = r'E:\PetImages\train'  # Path to your training data directory
test_dir = r'E:\PetImages\test'    # Path to your testing data directory

x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the CNN
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Extract features from CNN
extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
train_features = extractor.predict(x_train)
test_features = extractor.predict(x_test)

# Train SVM classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(train_features, y_train)

# Evaluate SVM classifier
svm_predictions = svm_classifier.predict(test_features)
print(classification_report(y_test, svm_predictions))

plt.plot(svm_predictions)
plt.show()


# In[ ]:




