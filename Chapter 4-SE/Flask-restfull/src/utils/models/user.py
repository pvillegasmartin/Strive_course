from utils.db import db
import uuid

class User(db.Model):
    user_id = db.Column(db.String(32), primary_key=True, default=str(uuid.uuid4()), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    surname = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())

    def to_json(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'surname': self.surname,
            'email': self.email,
            'created_at': str(self.created_at)
        }