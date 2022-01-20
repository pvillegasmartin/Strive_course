from envs.strive.Lib import distutils
from flask_restful import Resource
from flask import request
import uuid
from utils.models.todo import Todo
from utils.db import db


class HomeRoute(Resource):

    def get(self):
        todos = db.session.query(Todo).all()
        todos = [todo.to_json() for todo in todos]
        return {'data':todos}

    def post(self):

        title = request.form['title']
        description = request.form['description']
        done = eval(request.form['done'].title())
        todo = Todo(title=title, description=description, done=done)
        db.session.add(todo)
        db.session.commit()
        return {'data': todo.to_json()}


class HomeRouteWithId(Resource):
    def get(self, id):
        data_object = db.session.query(Todo).filter(Todo.id == id).first()
        if (data_object):
            return {'data':data_object.to_json()}
        else:
            return {'data':'Not Found'},404

    def put(self, id):
        data_object = db.session.query(Todo).filter(Todo.id == id).first()
        if (data_object):
            for key in request.form.keys():
                setattr(data_object, key, request.form[key])
            setattr(data_object, 'updated_at', db.func.now())
            db.session.commit()
            return {'data': data_object.to_json()}
        else:
            return {'data': 'Not Found'}, 404

    def delete(self, id):
        data_object = db.session.query(Todo).filter(Todo.id == id).first()
        if (data_object):
            db.session.delete(data_object)
            db.session.commit()
            return {'data':'DELETED'}
        else:
            return {'data': 'Not Found'}, 404