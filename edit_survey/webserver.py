from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/poll")
def poll():
    return "not yet implemented"


@app.route("/img/")
def img():
    return "not yet implemented"


if __name__ == "__main__":
    app.run()
