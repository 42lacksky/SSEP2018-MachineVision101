from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import re
import base64
import random
import pickle
from skimage.transform import resize
from model.model import MV101_KNN
from model.train import train_model
import os

data_path = './data/'
clf = train_model()

app = Flask(__name__)


@app.route('/')
@app.route('/prediction/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("prediction.html")

    image = request.form["img"]

    parse_image(image.encode('ascii'))

    x = imread(data_path + 'predictimage.png', mode='L')
    x = np.invert(x)
    x = resize(x, (28, 28))
    x = x.flatten() * 255

    prediction = clf.predict([x])
    print(prediction)

    return str(prediction)


@app.route('/markup/', methods=['GET', 'POST'])
def markup():
    if request.method == 'GET':
        return render_template("markup.html")

    image = request.form["img"]
    label = request.form["label"]

    parse_image(image.encode('ascii'), label)

    return str(label)


@app.route('/fine_tune/', methods=['GET', 'POST'])
def fine_tune():
    if request.method == 'GET':
        return render_template("finetune.html")

    image = request.form["img"]
    label = request.form["label"]

    x = imread(parse_image(image.encode('ascii'), label), mode='L')
    x = np.invert(x)
    x = resize(x, (28, 28))
    x = x.flatten() * 255

    global clf
    prediction = clf.predict([x])

    if int(prediction) == int(label):
        return "Я думаю это {} и это правильно! Думаю, учиться мне тут нечему.".format(label)

    clf = train_model()
    return "Эта цифра так похожа на {}, но это {}. Попробую больше так не ошибаться!".format(prediction, label)


def parse_image(image_data, label=None):
    image_str = re.search(b'base64,(.*)', image_data).group(1)

    if label is None:
        with open(data_path + 'predictimage.png', 'wb') as output:
            output.write(base64.decodebytes(image_str))

    file_path_to_save = data_path + str(label) + '_' + str(int(random.random() * 100000)) + '.png'
    with open(file_path_to_save, 'wb') as output:
        output.write(base64.decodebytes(image_str))

    print(os.path.abspath(file_path_to_save))
    return file_path_to_save


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
