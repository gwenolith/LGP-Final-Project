import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os

# Load your model
# Define your model architecture (same as the one used for training)
model = keras.models.load_model("new_plant_disease_prediction_model_transfer_learning.h5", compile=False)

# Then load weights
model.load_weights("new_plant_disease_prediction_model_transfer_learning.h5")
# Define image path
img_path = "uploads/b7400558-800px-wm.jpg"  # Change this to your actual image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # change size to your model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize if the model was trained that way

# Make prediction
prediction = model.predict(img_array)

print(prediction)