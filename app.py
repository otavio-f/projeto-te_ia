from blueprints import dataset
from blueprints import classifier
from blueprints import evaluator

from flask import Flask
from flask import render_template
from flask import redirect


app = Flask(__name__)
app.config['SERVER_NAME'] = '0.0.0.0:5000'


app.register_blueprint(dataset, url_prefix="/datasets")
app.register_blueprint(classifier, url_prefix="/classifiers")
app.register_blueprint(evaluator, url_prefix="/evaluators")


@app.route("/")
def home():
    return render_template("index.html.j2")


@app.route("/favicon.ico")
def icon():
    return redirect("/static/numpy_logo_icon_248343.ico", code=302)


if __name__ == "__main__":
    app.run(debug=True)