import cv2
import pyttsx3 as say
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess the new image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Perform blindness detection on a new image
def perform_blindness_detection(img_path):
    img = preprocess_image(img_path)
    features = model.predict(img)
    predicted_class = np.argmax(features)
    
    if predicted_class == 0:
        say.speak("The image indicates a normal eye.")
        print("The image indicates a normal eye.")
    else:
        say.speak("The image indicates an abnormal eye.")
        print("The image indicates an abnormal eye.")

# Path to the new image for blindness detection
new_image_path = "happy.jpg"

# Perform blindness detection on the new image
perform_blindness_detection(new_image_path)