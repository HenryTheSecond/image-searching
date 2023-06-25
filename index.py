from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import os
from test import FRUIT
from flask_cors import CORS, cross_origin

train_dir = "E:/dotnet/AI/fruit-360-new/fruits-360_dataset/fruits-360/Training"
class_names = os.listdir(train_dir)
model = tf.keras.models.load_model('E:/dotnet/AI/fruit-360-new/1')
app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route("/test-form", methods = ["POST"])
@cross_origin(supports_credentials=True)
def upload_file():
    imageRequest = request.files.get("images")
    img = PIL.Image.open(imageRequest)
    img = img.resize((100, 100))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return jsonify({"fruit": FRUIT[predicted_class]})


@app.route("/")
def home():
    for i in range(len(class_names)):
        print("'" +  class_names[i] + "'" + ": " + "''" + ",")
    return "OK"

if __name__ == '__main__':
    app.run()
