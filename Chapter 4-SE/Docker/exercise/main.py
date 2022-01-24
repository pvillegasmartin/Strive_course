import torch
from flask import Flask, render_template, request
from torchvision import transforms
import model
from PIL import Image
from torch import nn
from torchvision import datasets, transforms
from collections import OrderedDict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def run():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']
        img_path = './images/' + imagefile.filename
        imagefile.save(img_path)

        image = Image.open(img_path)
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        image.resize_(1, 784)

        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10
        mod = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_sizes[1], output_size)),
            ('softmax', nn.Softmax(dim=1))]))
        mod.load_state_dict(torch.load('model_mnist.pth'))
        mod.eval()
        ps = mod.forward(image)
        prediction = torch.max(ps, 1)[1].data.squeeze().item()

        return render_template('index.html', prediction = prediction)
    except:
        return render_template('index.html', prediction = 'You have to select a file')

if __name__ == '__main__':
    app.run(debug=True)