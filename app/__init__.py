from flask import Flask


def create_app():
    app = Flask(__name__)

    from app.routes import churn_api

    app.register_blueprint(churn_api)
    return app
