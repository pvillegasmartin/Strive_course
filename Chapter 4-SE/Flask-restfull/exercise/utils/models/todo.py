from utils.db import db
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Todo(db.Model):
    id = db.Column(db.String(32), primary_key=True, default=generate_uuid, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    done = db.Column(db.Boolean(), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())
    updated_at = db.Column(db.DateTime, nullable=False, default=db.func.now())

    def to_json(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'done': self.done,
            'created_at': str(self.created_at),
            'updated_at': str(self.updated_at)
        }