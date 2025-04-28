from server.controller import Controller

from flask import Blueprint
from flask import Response
from flask import request
from flask import render_template


__control = Controller.create()


#-| Endpoint Dataset: dataset |-----------------------------------------------

dataset = Blueprint("datasets", __name__)


@dataset.route("/")
def dataset_home() -> str:
    """
    Apresenta a página inicial de datasets.

    :returns: A página com a listagem de datasets.
    """
    return render_template("datasets/index.html.j2", datasets=__control.get_all_datasets())


@dataset.route("/new")
def dataset_form() -> str:
    """
    Apresenta a página que contém o formulário para criação de datasets.

    :returns: A página formatada com o formulário.
    """
    columns = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    return render_template("datasets/create-form.html.j2", columns=tuple(enumerate(columns)))


@dataset.route("/new", methods=["POST"])
def gen_dataset() -> dict:
    """
    Gera um novo conjunto de dados.

    :returns: O id do conjunto de dados criado.
    """
    train_percent = float(request.json.get("train_percent", "0.7"))
    columns = request.json.get("columns", ["Petal length", "Petal width"])
    if len(columns) == 0:
        return Response("Missing parameters!", 400)
    response = __control.gen_dataset(train_percent=train_percent, columns=columns)
    return {"id": response}


@dataset.route("/info/<dsid>")
def get_dataset(dsid: str):
    """
    Retorna a página de detalhes e operações para o dataset.
    :param dsid: O id do dataset.
    :returns: Dados do conjunto de dados, ou 404 se não existir.
    """
    response = __control.get_dataset(dsid)

    if response is None:
        return render_template('datasets/not-found.html.j2', name=dsid)
    
    return render_template('datasets/info.html.j2', name=dsid, data=response)
    

#-| Endpoint Classificadores: classifier/ |-----------------------------------------

classifier = Blueprint("classifiers", __name__)

@classifier.route("/")
def classifier_home():
    """
    Apresenta a página inicial de classificadores.
    """
    return render_template("classifiers.html.j2")

@classifier.route("/min")
def euclidean_dist():
    dsid = request.args["id"]
    try:
        return render_template(
            "classifiers/euclideandist-result.html.j2",
            data=__control.euclidean_distance(dsid), dataset=dsid)
    except KeyError:
        return render_template('datasets/not-found.html.j2', name=dsid)

@classifier.route("/max")
def maximum():
    dsid = request.args["id"]
    try:
        return render_template(
            "classifiers/maximum-result.html.j2",
            data=__control.maximum(dsid), dataset=dsid)
    except KeyError:
        return render_template('datasets/not-found.html.j2', name=dsid)

@classifier.route("/dij")
def dij():
    dsid = request.args["id"]
    try:
        return render_template(
            "classifiers/dij-result.html.j2",
            data=__control.dij(dsid), dataset=dsid)
    except KeyError:
        return render_template('datasets/not-found.html.j2', name=dsid)


@classifier.route("/perceptron")
def perceptron():
    dsid = request.args["id"]
    try:
        return render_template(
            "classifiers/perceptron-result.html.j2",
            data=__control.perceptron(dsid, 10_000), dataset=dsid)
    except KeyError:
        return render_template('datasets/not-found.html.j2', name=dsid)


@classifier.route("/perceptron_delta")
def perceptron_delta():
    dsid = request.args["id"]
    try:
        return render_template(
            "classifiers/perceptron-delta-result.html.j2",
            data=__control.perceptron_delta(dsid, 10_000), dataset=dsid)
    except KeyError:
        return render_template('datasets/not-found.html.j2', name=dsid)



#-| Endpoint avaliadores: /evaluators |----------------------------------------

evaluator = Blueprint("evualuators", __name__)
