from server.control import Controller
from server.errors import InvalidAlgorithm, InvalidClass, InvalidDataset

from flask import Blueprint
from flask import Response
from flask import request
from flask import render_template
from flask import redirect

from dataclasses import dataclass


__control = Controller()


# TODO: Retornar templates com os dados já construídos

#-| Endpoint Dataset: dataset |-----------------------------------------------

dataset = Blueprint("datasets", __name__)


@dataset.route("/", methods=["GET"])
def dataset_home():
    """
    Apresenta a página inicial de datasets
    """
    return render_template("datasets/index.html.j2", datasets=__control.get_datasets())


@dataset.route("/new", methods=["GET"])
def dataset_form():
    """
    Apresenta a página que contém o formulário para criação de datasets
    """
    columns = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    return render_template("datasets/create-form.html.j2", columns=tuple(enumerate(columns)))


@dataset.route("/new", methods=["POST"])
def gen_dataset():
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


@dataset.route("/all", methods=["GET",])
def all_datasets():
    """
    Retorna a lista de todos os conjuntos de dados criados.
    :returns: Lista dos conjuntos de dados.
    """
    try:
        response = __control.get_datasets()
        return response
    except IOError: # Erro temporário
        return Response("Internal error", status=500, mimetype="text/plain")


@dataset.route("/info/<dsid>", methods=["GET"])
def get_dataset(dsid: str):
    """
    Retorna a página de detalhes e operações para o dataset.
    :param dsid: O id do dataset.
    :returns: Dados do conjunto de dados, ou 404 se não existir.
    """
    try:
        response = __control.get_dataset(dsid)
        return render_template('datasets/info.html.j2', name=dsid, data=response)
    except InvalidDataset:
        return render_template('datasets/not-found.html.j2', name=dsid)
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

#-| Endpoint Classificadores: classifier/ |-----------------------------------------

classifier = Blueprint("classifiers", __name__)

@classifier.route("/", methods=["GET"])
def classifier_home():
    """
    Apresenta a página inicial de classificadores.
    """
    return render_template("classifiers.html.j2")

@classifier.route("/eucdist")
def euclidean_dist():
    try:
        dsid = request.args["id"]
        # response = controller.euclidean_distance(dsid, "setosa")
        result = __control.euclidean_distance2(dsid)
        return render_template('classifiers/euclideandist-result.html.j2', data=result)
    except InvalidDataset:
        return render_template('datasets/not-found.html.j2', name=dsid)
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")


@classifier.route("/max", methods=["POST",])
def maximum(dsid: str):
    try:
        dsid = request.json["id"]
        class_name = request.json["class"]
        response = __control.maximum(dsid, class_name)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@classifier.route("/dij", methods=["POST",])
def dij(dsid: str):
    try:
        dsid = request.json["id"]
        c1, c2 = request.json["classes"]
        response = __control.dij(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@classifier.route("/perceptron", methods=["POST",])
def perceptron(dsid: str):
    try:
        dsid = request.json["id"]
        c1, c2 = request.json["classes"]
        response = __control.perceptron(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@classifier.route("/perceptron_delta", methods=["POST",])
def perceptron_delta(dsid: str):
    raise NotImplementedError
    
    try:
        dsid = request.json["id"]
        c1, c2 = request.json["classes"]
        response = __control.perceptron_delta(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")

#-| Endpoint avaliadores: /evaluators |----------------------------------------

evaluator = Blueprint("evualuators", __name__)
