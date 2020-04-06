import torch

from PIL import Image
from flask import Flask, request, render_template

from commons import get_model
from detect import plot_boxes, detect
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(device)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        img_name = file.filename
        if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
            res_img = 'dect_' + file.filename
            img = Image.open(file)
            if img_name.endswith('png'):
                img = img.convert('RGB')
            boxes = detect(img, model, device)
            plot_img = plot_boxes(img, boxes)
            plot_img.save('static/' + res_img, optimize=True)
            return render_template('result.html', img='static/' + res_img)
        else:
            print('Only img file')
            return


if __name__ == '__main__':
    app.run(debug=True)
