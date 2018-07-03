# Import all flask related functions
from flask import Flask, render_template, session, jsonify

# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

# Import flask_mail
from flask_mail import Mail, Message

# Import wraps for authentication of users
from functools import wraps

# Define the WSGI application object
app = Flask(__name__, static_url_path="/static")

# Configuring the static folder
app.config['STATIC_FOLDER'] = 'static'

# Configurations
app.config.update(
	DEBUG=True,
	MAIL_SERVER='smtp.gmail.com',
	MAIL_PORT=465,
	MAIL_USE_SSL=True,
	MAIL_USERNAME='7d.shock@gmail.com',
	MAIL_PASSWORD='dshock343'
	)
mail = Mail(app)

# Configurations
app.config.from_object('config')

# Configurations for SQLAlchemy
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Define the database object which is imported
# by modules and controllers
db = SQLAlchemy(app)

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
   return render_template('index.html'), 200

# Authentication functionality
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return jsonify(message="Unauthorized", success=False), 401
        return f(*args, **kwargs)
    return decorated

# Import a module / component using its blueprint handler variable (mod_auth)
#ONLY HAVE TO ADD LINES TO NEXT TO PARAS AS WE ADD OBJECTS
from app.server.api.controllers import mod_models

# Register blueprint(s)
app.register_blueprint(mod_models)

# Build the database:
# This will create the database file using SQLAlchemy
db.create_all()
