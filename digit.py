import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

#print(tf.__version__)

path = 'mnist.npz'

# get data - this will be cached 
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)

"""like we do in other models we do not have to downlaod the dataset here
it is fetched automatically using the Tensor flow keras API"""

print(x_train.shape)
print(x_test.shape)

"""
#it is plotting the cure and fetching the  images from those 60K
#images fetched from that API
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], aspect=1, cmap='gray')
plt.show()

#this is also doing the same and it is showing the 16 images from
#those 10K images

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[i], aspect=1, cmap='gray')
plt.show()
"""


# set up TF model and train 

# callback 

# if model accuracy goes beyond 99% then after completing that 
# epoch it stops the training
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    print(logs)
    if(logs.get('accuracy') > 0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True


callbacks = myCallback()
"""
# normalise 
x_train, x_test = x_train/255.0, x_test/255.0

# create model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# fit model
history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# stats 
print(history.epoch, history.history['accuracy'][-1])"""

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model2.summary())

# fit model
history2 = model2.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# stats 
print(history2.epoch, history2.history['accuracy'][-1])

# we are taking some data from the dataset lists and making probabilities
# using predict function

# Load your custom image
custom_image = Image.open('image.png')
custom_image = custom_image.convert('L')

# Resize the image to match the input size expected by your model
# Assuming your model expects images of size (28, 28)
custom_image = custom_image.resize((28, 28))

# Convert the image to a numpy array and normalize the pixel values
custom_image_array = np.array(custom_image) / 255.0

# Expand the dimensions of the image array to match the input shape expected by your model
custom_image_array = np.expand_dims(custom_image_array, axis=0)

# Make predictions on the custom image
res = model2.predict(custom_image_array)

print("\nRes list of probabilities\n")
res = model2.predict(custom_image_array)                #(x_test[11:12])
print(res)

# after adding softmax we are finding the index corresponding 
# to that probility

print("\nafter adding softmax\n")
probability_model = tf.keras.Sequential([model2, tf.keras.layers.Softmax()])
res = probability_model.predict(custom_image_array)                         #(x_test[11:12])
index = np.argmax(res)
print(res)
print(index)

"""
# display the digit images we created 
#img_dir = './Digits/'
img_names = ['0.png', '1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png']
imgs = [np.array(Image.open('./Digits/'+img_name)) for img_name in img_names]

plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(imgs[i], aspect=1, cmap='gray')
plt.show()

# our own prediction on some of the sample images
def predict():
    img_data = np.array(imgs)
    img_data = np.expand_dims(img_data, axis=-1)  # Add channel dimension
    res = probability_model.predict(img_data)
    print([np.argmax(a) for a in res])
predict()"""