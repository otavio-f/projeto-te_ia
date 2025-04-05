from flask import Flask
from flask import Response
from flask import request

from server.control import Controller
from server.control import InvalidClass
from server.control import InvalidDataset


app = Flask(__name__)
controller = Controller()


@app.route("/", methods=["GET",])
def home():
    return Response("Hello, world!", status=200, mimetype="application/json")


@app.route("/dataset/<string:dsid>/eucdist", methods=["POST",])
def euclidean_dist(dsid: str):
    try:
        class_name = request.json["class"]
        response = controller.euclidean_distance(dsid, class_name)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")


@app.route("/dataset/<string:dsid>/max", methods=["POST",])
def maximum(dsid: str):
    try:
        class_name = request.json["class"]
        response = controller.maximum(dsid, class_name)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@app.route("/dataset/<string:dsid>/dij", methods=["POST",])
def dij(dsid: str):
    try:
        c1, c2 = request.json["classes"]
        response = controller.dij(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@app.route("/dataset/<string:dsid>/perceptron", methods=["POST",])
def perceptron(dsid: str):
    try:
        c1, c2 = request.json["classes"]
        response = controller.perceptron(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")
    # except:
    #     return Response("Internal error", status=500, mimetype="text/plain")

@app.route("/dataset/<string:dsid>/perceptron_delta")
def perceptron_delta(dsid: str):
    raise NotImplementedError
    
    try:
        c1, c2 = request.json["classes"]
        response = controller.perceptron_delta(dsid, c1, c2)
        return response
    except InvalidDataset:
        return Response("Dataset Not Found", status=404, mimetype="text/plain")
    except InvalidClass:
        return Response("Invalid class", status=400, mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)