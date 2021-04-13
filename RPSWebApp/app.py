import numpy as np
import os
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
import tensorflow as tf
global graph
from tensorflow.python.keras.backend import set_session
sess = tf.compat.v1. Session()
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
set_session(sess)
model = load_model(r".\models\rps.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            json_file = open('models/rps.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("models/rps.h5")      
            loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            preds = loaded_model.predict_classes(x)
            print("prediction",preds)
            
        index = ['rock','paper','scissors']
        
        text = index[preds[0]]
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        