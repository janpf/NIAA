from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/poll")
def poll():
    return "not yet implemented"


@app.route("/img/")
def poll():
    return "not yet implemented"


if __name__ == "__main__":
    app.run()
