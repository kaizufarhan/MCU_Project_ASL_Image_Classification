"""
    For REU Deeper Dive
    Train a neural network model to classify images of clothing
    Created by Adam Kohl
"""
# Import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions to be used further down in the code
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Import Fashion MNIST dataset
# 70,000 grayscale images (28 x 28) in 10 categories
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale values to a range of 0 - 1 before feeding them to Neural Network model
# Each pixel has a value of 255 because of RGB colors. Since we want it from 0 to 1, if we divide by 255, we get a percentage between 0-1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verify the dataset is in the correct format
# Display first 25 images from the training set and display the class label
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model
# A layer is a basic building block
#   - Extracts representations from data fed into them
# Deep learning has more than 1 layer chained together
model = tf.keras.Sequential()

# Add the first layer defining the image input shape
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Add a layer of 128 fully-connected neurons (FCN)
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add output layer of 10 fully-connected neurons (FCN)
# Each neuron corresponds to a class name
model.add(tf.keras.layers.Dense(10))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
# Fit function is 'fitting' the model to the training data
model.fit(train_images, train_labels, epochs=10)

# Compare accuracy on test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
# Add a softmax layer to the model to convert linear outputs to probabilities
# Makes it easier to interpret
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])

# Identify the label with the highest confidence value
print(np.argmax(predictions[0]))

# Verify predictions
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Use the trained model to make a prediction
# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)

# Make a prediction
predictions_single = probability_model.predict(img)
print(predictions_single)

# See the results
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))