from flask import Flask, render_template, request
from flask_restful import Resource, Api
from routes.home.route import HomeRoute, HomeRouteWithId
from utils.db import db
import requests
import json

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db.sqlite"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    api = Api(app)

    db.init_app(app) #initialize the database
    db.create_all(app=app) #create tables

    api.add_resource(HomeRoute, '/')
    api.add_resource(HomeRouteWithId, '/<string:id>')
    return app



if __name__ == '__main__':
    app = create_app()

    @app.route('/home', methods=['GET'])
    def home():
        req = requests.get('http://127.0.0.1:5000/')
        data = json.loads(req.content)
        return render_template('home.html', data=data)


    @app.route("/add", methods=["GET","POST"])
    def show_signup_form():
        if request.method == 'POST':
            title = request.form.get('title')
            description = request.form.get('description')
            done = request.form.get('done')
            data = {'title':title, 'description':description, 'done':done}
            requests.post('http://127.0.0.1:5000/',data=data)

        return render_template("add.html")


    @app.route("/delete", methods=["GET","POST","DELETE"])
    def show_delete():
        if request.method == 'POST':
            id = request.form.get('id')
            requests.delete(f'http://127.0.0.1:5000/{id}')
        return render_template("delete.html")

    app.run(debug=True)