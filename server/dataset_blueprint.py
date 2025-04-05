from server.control import Controller

from flask import Blueprint
from flask import Response
from flask import Request


app = Blueprint("datasets", url_prefix="/dataset")
__control = Controller()


@app.route("/", methods=["GET",])
def all_datasets():
    """
    Retorna a lista de todos os conjuntos de dados criados.
    :returns: Lista dos conjuntos de dados.
    """
    try:
        response = __control.get_datasets()
        return response
    except IOError:
        return Response("Internal error", status=500, mimetype="text/plain")


@app.route("/", methods=["POST",])
def gen_dataset():
    """
    Gera um novo conjunto de dados.
    :returns: O id do conjunto de dados criado.
    """
    train_percent = float(request.json.get("train_percent", "0.7"))
    columns = request.json.get("columns", ["Petal length", "Petal width"])
    response = __control.gen_datasets(train_percent=train_percent, columns=columns)
    return {"id": response}


@app.route("/dataset/<string:dsid>", methods=["GET",])
def get_dataset(dsid: str):
    """
    Retorna o conjunto de dados criado.
    :returns: Dados do conjunto de dados, ou 404 se não existir.
    """
    try:
        response = __control.get_dataset(dsid)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")
