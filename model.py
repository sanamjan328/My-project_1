# coding=utf-8
from flask import Flask, render_template, request, send_file, send_from_directory, jsonify, json, redirect, url_for
from flask_cors import cross_origin
from PIL import Image, ImageOps
import random, datetime
from helpers_cntk import *
from cntk import load_model, combine
from helpers import *
distMethod = "weightedL2"
from io import *
from werkzeug import secure_filename
from json import dumps
import os, sys

__author__ = 'sanam'

app = Flask(__name__)
rootDir = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\", "/") + "/"
datasetName = "Furnitures"
procDir = rootDir + "proc/" + datasetName + "/"
imgDir = rootDir + "data/" + datasetName + "/"
APP_ROOT = os.path.dirname(__file__)
RESULTS_ARRAY = []

app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'JPG'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'js_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/js', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    elif endpoint == 'css_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/css', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/css/<path:filename>')
def css_static(filename):
    return send_from_directory(app.root_path + '/static/css/', filename)

@app.route('/js/<path:filename>')
def js_static(filename):
    return send_from_directory(app.root_path + '/static/js/', filename)

@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    filename = app.root_path + '/data/Furnitures/' + filename
    with open(filename, 'rb') as f:
        data = io.BytesIO(f.read())
    return send_file(data, attachment_filename=filename,
                     mimetype='image/{}'.format(get_file_type(filename)))
# def data_static(filename):
#  return send_from_directory(app.root_path + '/data/Furnitures/', filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        files = request.files['file']
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            #app.logger.info('FileName: ' + filename)
            updir = os.path.join(rootDir, 'data/Furnitures/NewData')
            files.save(os.path.join(updir, filename))
            # file_size = os.path.getsize(os.path.join(updir, filename))
            random.seed(0)
            printDeviceType()
            # print("basla")
            imageFilePath = os.path.join(updir, filename)
            queryImg = imread(imageFilePath) # Image.open(imageFilePath)

            # cntk load #######################################################

            cntkRefinedModelPath = procDir + "cntk.model"
            model = load_model(cntkRefinedModelPath)
            node = model.find_by_name("poolingLayer")
            model = combine([node.owner])

            # svm load
            svmpath = procDir + "svm.np"
            svmLearner = loadFromPickle(svmpath)
            svmBias = svmLearner.base_estimator.intercept_
            svmWeights = np.array(svmLearner.base_estimator.coef_[0])

            # load image reference features
            imgInfosTPath = procDir + "imgInfosTest.pickle"
            featuresPath = procDir + "features.pickle"
            refImgInfos = loadFromPickle(imgInfosTPath)
            ImageInfo.allFeatures = loadFromPickle(featuresPath)

            # computation
            rf_inputResoluton = 224
            imgPadded = imresizeAndPad(queryImg, rf_inputResoluton, rf_inputResoluton, padColor=[114, 0, 0])
            arguments = {
                model.arguments[0]:
                    [np.ascontiguousarray(np.array(imgPadded, dtype=np.float32).transpose(2, 0, 1))]
            }

            # run dnn model
            dnnOut = model.eval(arguments)
            queryFeat = np.concatenate(dnnOut, axis=0).squeeze()
            queryFeat = np.array(queryFeat, np.float32)

            # will be compute distances between query image and all other images
            svm_boL2Normalize = True
            print("Distance computation using {} distance.".format(distMethod))
            dists = []
            for refImgInfo in refImgInfos:
                refFeat = refImgInfo.getFeat()
                dist = computeVectorDistance(queryFeat, refFeat, distMethod, svm_boL2Normalize, svmWeights, svmBias,
                                             svmLearner)
                dists.append(dist)

            # result print
            sortOrder = np.argsort(dists)
            if distMethod.lower().endswith('prob'):
                sortOrder = sortOrder[:40]
            sortOrder = sortOrder[::-1]
            for index, refIndex in enumerate(zip(sortOrder)):
                refIndex = int(refIndex[0])
                # refIndex = int(refIndex[0])
                currDist = dists[refIndex]
                refImgPath = refImgInfos[refIndex].getImgPath(imgDir)
                # listImg = imread(imconvertPil2Numpy(imread(refImgPath)))
                # ID = format(index, currDist)
                RESULTS_ARRAY.append({"image": refImgPath.replace(imgDir,'images/')})

            # print("nanik")
            results =  (RESULTS_ARRAY[::-1][:40])
            # json_data = dumps(results,  indent=2) # jsonify(results)
            return jsonify(results)


