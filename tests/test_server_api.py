import unittest
import requests
from flask import Flask
from multiprocessing import Process
from blueprints import dataset, classifier, evaluator
import time


class ServerTestCase(unittest.TestCase):
    """
    Classe de testes de unidade da api do servidor.
    """
    base_url = "http://localhost:5000/"

    def setUp(self):
        app = Flask(__name__)
        app.config['SERVER_NAME'] = f'localhost:5000'
        app.register_blueprint(dataset)
        app.register_blueprint(classifier)
        app.register_blueprint(evaluator)
        self.app = Process(target=app.run)
        self.app.start()


    def tearDown(self):
        self.app.terminate()
        time.sleep(2)

    def testGenDataset(self):
        """Testa geração de dataset."""
        params = {"train_percent": 0.5, "columns": ["Petal length", "Petal width"]}
        request = requests.post(self.base_url+"dataset/", json=params, timeout=5)

        self.assertIn("id", request.json())
        self.assertEqual(request.status_code, 200)
        

    def testGetDatasetsEmpty(self):
        """Testa obtenção de todos datasets sem ter criado nenhum."""
        request = requests.get(self.base_url+"dataset/all", timeout=5)

        self.assertEqual([], request.json())
        self.assertEqual(request.status_code, 200)
        

    def testGetDatasets(self):
        """Testa criação e obtenção de datasets."""
        params = {"train_percent": 0.5, "columns": ["Petal length", "Petal width"]}
        requests.post(self.base_url+"dataset/", json=params, timeout=5)

        params = {"train_percent": 0.75, "columns": ["Petal length", "Sepal length"]}
        requests.post(self.base_url+"dataset/", json=params, timeout=5)

        request = requests.get(self.base_url+"dataset/all", timeout=5)

        self.assertEqual(request.status_code, 200)
        self.assertEqual(2, len(request.json()))
    

    def testEuclideanDistance(self):
        """Testa distância euclideana."""
        params = {"train_percent": 0.75, "columns": ["Petal length", "Petal width"]}
        request = requests.post(self.base_url+"dataset/", json=params, timeout=5)
        dataset = request.json()["id"]

        params = {"train_percent": 0.75, "columns": ["Petal length", "Sepal length"]}
        requests.post(self.base_url+"dataset/", json=params, timeout=5)

        request = requests.get(self.base_url+"dataset/all", timeout=5)

        self.assertEqual(request.status_code, 200)
        self.assertEqual(2, len(request.json()))
    
