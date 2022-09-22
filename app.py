from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from utils import *
from keras.models import load_model
import numpy as np
from cv2 import imread, resize
from PIL import Image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UNET_FOLDER'] = os.path.join(APP_ROOT, 'static/uploads/unet')
app.config['VGG_FOLDER'] = os.path.join(APP_ROOT, 'static/uploads/vgg')

@app.route("/")
def hello_world():
    tpyeTest = request.args.get('type')
    listFiles = os.listdir('static/uploads')
    return render_template('index.html', type_test=tpyeTest, listFiles=listFiles, BASE_DIR=app.config['UPLOAD_FOLDER'])

@app.route("/simple", methods=['GET', 'POST'])
def simpleTest():
    tpyeTest = 'simple'
    f = request.files['filesImage']
    filename = secure_filename(f.filename)
    filename = str(filename).split('.')
    filename = 'citra.' + filename[-1]
    loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(loc)
    segmentation(loc)
    segmentation(loc, typec='B')
    savePredMask(loc)
    savePredMask(loc, invert=True)

    listFiles = os.listdir('static/uploads')
    listPred = classification()
    
    return render_template('index.html', type_test=tpyeTest, listFiles=listFiles, listPred=listPred)

@app.route("/advance", methods=['GET', 'POST'])
def advanceTest():
    type_test = 'advance'
    type_model = request.form['modelSelect']
    print('ini type model : ',type_model)

    f = request.files['filesImage']
    filename = secure_filename(f.filename)
    filename = str(filename).split('.')
    filename = 'citra.' + filename[-1]

    if type_model == 'unet' :
        loc = os.path.join(app.config['UNET_FOLDER'], filename)
        f.save(loc)
        segmentation(loc, typec='A', dir_default='uploads/unet')
        segmentation(loc, typec='B', dir_default='uploads/unet')
        savePredMask(loc, invert=False, dir_default='uploads/unet')
        savePredMask(loc, invert=True, dir_default='uploads/unet')
        return render_template('index.html', type_test=type_test, type_model=type_model)
    else:
        loc = os.path.join(app.config['VGG_FOLDER'], filename)
        f.save(loc)
        resultPred = classification(imgFileArray=loc, all=False)
        print(resultPred)
        return render_template('index.html', type_test=type_test, type_model=type_model, resultPred=resultPred)

def segmentation(imgFile, typec = 'A', dir_default='uploads'):
    img = imread(imgFile)
    img = resize(img, (224,224))
    img = img.reshape(1,224,224,3)

    modelPath = 'static/model/result-model_dua_temp.h5'
    model = load_model(modelPath, compile=False)
    result = model.predict(img)

    if typec == 'A' :
        cutA = cutArea(img, result)
        saveA = cutA.reshape(224,224,3)
        Image.fromarray(saveA).convert("RGB").save(f"static/{dir_default}/img_cut_a.png")
    else :
        cutB = cutAreaInverted(img, result)
        saveB = cutB.reshape(224,224,3)
        Image.fromarray(saveB).convert("RGB").save(f"static/{dir_default}/img_cut_b.png")

def classification(imgFileArray=None, all=True):

    files = ['citra.png','img_cut_a.png', 'img_cut_b.png']
    target_class = ['COVID', 'Normal', 'Pneumonia']
    modelPath = 'static/model/vgg19_fold-min-5.h5'
    model = load_model(modelPath)
    predDict = dict()

    if all :
        for i in files:
            img = resize(imread('static/uploads/' + i), (224,224))
            img = img.reshape(1,224,224,3)
            res = model.predict(img)
            predDict[i] = target_class[np.argmax(res)]
        
        return predDict
    else:
        img = resize(imread(imgFileArray), (224,224))
        img = img.reshape(1,224,224, 3)
        res = model.predict(img)
        
        return target_class[np.argmax(res)]

def savePredMask(imgFile, invert=False, dir_default='uploads'):
    img = imread(imgFile)
    img = resize(img, (224,224))
    img = img.reshape(1,224,224,3)

    modelPath = 'static/model/result-model_dua_temp.h5'
    model = load_model(modelPath, compile=False)
    result = model.predict(img)
    if not invert :
        savePredMask = makeImageMask(result)
        savePredMask = savePredMask.reshape(224,224,3)
        Image.fromarray(savePredMask).convert("RGB").save(f"static/{dir_default}/mask_predMask.png")
    else:
        savePredMask = makeImageMaskInverted(result)
        savePredMask = savePredMask.reshape(224,224,3)
        Image.fromarray(savePredMask).convert("RGB").save(f"static/{dir_default}/mask_predMaskInverted.png")