from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def run():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    prediction = 0
    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)