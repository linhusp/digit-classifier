from digit_classifier import (
    KerasImgClassifier,
    normalize_img_data
)
from flask import (
    request,
    Flask,
    render_template
)
from PIL import Image
import re
import base64
import numpy as np

app = Flask(__name__)

global model, model_path
# load pretrained model
model_path = './model/model.h5'
model = KerasImgClassifier(model_path, input_dim=None, nlabels=None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    file_name = 'output.jpg'
    # raw data
    img_data = request.get_data()
    # decode image
    img_str = re.search(r'base64,(.*)', str(img_data)).group(1)
    with open(file_name, 'wb') as f:
        f.write(base64.b64decode(img_str))
    # load image and convert it to grayscale
    img = Image.open(file_name).convert('L')
    # resize image to 28x28
    img.thumbnail((28, 28))
    # create input data from img
    X = np.array([np.asarray(img, dtype='int32')])
    # reshape input to tensor [1 28 28 1]
    X = X.reshape(X.shape + (1,))
    # normalize input
    X = normalize_img_data(X)

    # classify input
    model.input_dim = X[0, :].shape
    model.nlabels = 10
    y_pred = model.classify(X)
    return str(y_pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
