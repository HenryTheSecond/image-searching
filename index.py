from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import os
#from test import FRUIT
from flask_cors import CORS, cross_origin
from constant import TRAIN_DIR, MODEL_DIR
import json
import time

train_dir = TRAIN_DIR
class_names = os.listdir(train_dir)
model = tf.keras.models.load_model(MODEL_DIR)
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
    labels = read_data()
    return jsonify({"fruit": labels[predicted_class]})


@app.route("/get-labels", methods = ["GET"])
@cross_origin(supports_credentials=True)
def get_labels():
    labels = read_data()
    labelArray = []
    for label in labels.keys():
        labelArray.append({"en": label, "vi": labels[label]})
    return jsonify(labelArray)


@app.route("/add-images", methods = ["POST"])
@cross_origin(supports_credentials=True)
def add_images():
    imgs = request.files.getlist("images")
    labelRequest = json.loads(request.form.get('label'))
    labels = read_data()
    if(labels.__contains__(labelRequest['en'].lower()) == False):
        path = os.path.join(train_dir, labelRequest['en'])
        os.makedirs(path)
        if(labelRequest.__contains__('vi') == False or labelRequest['vi'] == ''):
            labelRequest['vi'] = labelRequest['en']
        write_data(labelRequest['en'], labelRequest['vi'])
    for img in imgs:
        imgOpen = PIL.Image.open(img)
        imgOpen = imgOpen.resize((100, 100))
        path = os.path.join(train_dir, labelRequest['en'], str(int(time.time())) + '_' + img.filename + '.jpg')
        imgOpen.save(path)
    return jsonify({'message': 'OK', 'status': 200})
    #return os.path.join(train_dir, 'aa', str(int(time.time())) + '.jpg')
    
    

def read_data():
    label = {}
    with open('label.txt', encoding='UTF-8') as myfile:
        for line in myfile.readlines():
            arg = line.replace('\n','').split(':')
            label[arg[0].lower()] = arg[1]
    return label

def write_data(en, vi):
    with open('label.txt', 'a', encoding='UTF-8') as myfile:
        myfile.write(en + ':' + vi + '\n')

@app.route("/")
def home():
    for i in range(len(class_names)):
        print("'" +  class_names[i] + "'" + ": " + "''" + ",")
    return "OK"

if __name__ == '__main__':
    app.run()
