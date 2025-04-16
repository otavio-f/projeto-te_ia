import random

from dataclasses import dataclass
from dataclasses import field

from vector.dataset import DataExtractor
from vector.dataset import DataSet
import classifier.classifiers as classifier

from server.errors import InvalidAlgorithm, InvalidClass, InvalidDataset


@dataclass
class Controller(object):
    """Controlador da aplicação."""

    datasets: dict = field(default_factory=dict)

    @property
    def __new_id(self) -> str:
        """
        Cria um novo id.
        :returns: um id de 8 algarismos hexadecimais aleatórios.
        """
        new_id = random.choices("abcdef1234567890", k=8)
        return "".join(new_id)

    def gen_dataset(self, train_percent: float, columns: [str, ...], src="data/Iris.csv"):
        """
        Gera um conjunto de dados.
        :param train_percent: A porcentagem de dados a serem separados para treino
        :param columns: As colunas selecionadas
        :param src: A fonte de dados, por padrão o conjunto de dados iris
        :returns: O id do conjunto de dados criado.
        """
        extractor = DataExtractor(src)
        conversor = lambda x: float(x.replace(",", "."))
        se_train, se_test = extractor.extract_by_class("Species", "setosa", columns, conversor).get_training_data(train_percent)
        ve_train, ve_test = extractor.extract_by_class("Species", "versicolor", columns, conversor).get_training_data(train_percent)
        vi_train, vi_test = extractor.extract_by_class("Species", "virginica", columns, conversor).get_training_data(train_percent)
        
        set_id = self.__new_id
        self.datasets[set_id] = {
            "info": {
                "columns": columns,
                "train_percent": train_percent
            },
            "dataset": {
                "setosa": {"train": se_train, "test": se_test},
                "versicolor": {"train": ve_train, "test": ve_test},
                "virginica": {"train": vi_train, "test": vi_test},
            }
        }

        return set_id
    
    def get_datasets(self) -> dict|None:
        """
        Recupera informação sobre todos os conjuntos de dados.
        :returns: Os conjuntos de dados.
        """
        return [
            {key: data["info"]}
            for key, data in self.datasets.items()
        ]


    def get_dataset(self, dataset_id: str) -> dict|None:
        """
        Recupera um conjunto de dados com o id especificado.
        :returns: O conjunto de dados ou None se não existir.
        """
        if not dataset_id in self.datasets:
            raise InvalidDataset

        result = self.datasets[dataset_id.lower()]["dataset"]
        info = self.datasets[dataset_id.lower()]["info"]
        # todo: retornar dataset inteiro
        return {
            "info":info,
            "dataset": {
            c: {
                "m" : result[c]["train"].m,
                "test" : list(result[c]["test"].lines)
            }
            for c in ("setosa", "versicolor", "virginica")}
        }


    def calc_euc(self, dataset: dict, class_name: str) -> dict:
        """
        Calcula a distância euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        """
        m = dataset[class_name]["train"].m

        func, eq = classifier.euclidean_dist(m)
        results = {
            c : [{"class": c, "data": sample, "result": func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        return {
            "equation": eq,
            "result": results
        }

    def euclidean_distance2(self, dataset_id: str) -> dict:
        """
        Calcula a distância euclideana do conjunto de dados de teste e as classes.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se o nome da classe não for válido
        """
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        
        dataset = self.datasets[dataset_id.lower()]["dataset"]
        classes = ("setosa", "versicolor", "virginica")

        dists = dict()
        for c in classes:
            func, eq = classifier.euclidean_dist(dataset[c]["train"].m)
            dists[c] = {"function": func, "equation": eq}
        
        results = [{
            "original_class": c,
            "data": sample,
            "result_class": min(classes, key=lambda c: dists[c]["function"](sample))
            }
            for c in classes
            for sample in dataset[c]["test"].lines
        ]

        results = {
            "equations": {
                c: dists[c]["equation"]
                for c in classes
            },
            "results": results
        }

        return results


    def euclidean_distance(self, dataset_id: str, class_name: str) -> dict:
        """
        Calcula a distância euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se o nome da classe não for válido
        """
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        if not class_name in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")

        dataset = self.datasets[dataset_id.lower()]["dataset"]
        m1 = dataset[class_name]["train"].m

        func, eq = classifier.euclidean_dist(m1)
        results = {
            c : [{"class": c, "data": sample, "result": func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        return {
            "equation": eq,
            "result": results
        }


    def maximum(self, dataset_id: str, class_name: str) -> dict:
        """
        Calcula a distância euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se o nome da classe não for válido
        """
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        if not class_name in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")

        dataset: DataSet = self.datasets[dataset_id.lower()]["dataset"]
        m1 = dataset[class_name]["train"].m

        func, eq = classifier.max_dist(m1)
        results = {
            c: [{"data": sample, "result":func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        # results = {
        #     c : [{"data": sample, "result": func(sample)} for sample in zip(*dataset[c]["test"].columns)]
        #     for c in ("setosa", "versicolor", "virginica")
        # }

        return {
            "eq": eq,
            "test_result": results
        }


    def dij(self, dataset_id: str, c1: str, c2: str) -> 'Equation':
        """
        Calcula a superfície de decisão euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se o nome da classe não for válido
        """
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        if not c1 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")
        if not c2 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")

        dataset = self.datasets[dataset_id.lower()]["dataset"]
        m1 = dataset[c1]["train"].m
        m2 = dataset[c2]["train"].m

        func, eq = classifier.dij(m1, m2)
        results = {
            c : [{"data": sample, "result": func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        return {
            "eq": eq,
            "test_result": results
        }


    def perceptron(self, dataset_id: str, c1: str, c2: str) -> dict:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se o nome da classe não for válido
        """
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        if not c1 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")
        if not c2 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")

        dataset = self.datasets[dataset_id.lower()]["dataset"]
        c1v = dataset[c1]["train"].lines
        c2v = dataset[c2]["train"].lines

        func, eq, iters = classifier.perceptron(c1v, c2v)
        results = {
            c : [{"data": sample, "result": func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        return {
            "eq": eq,
            "iters": iters,
            "test_result": results
        }


    def perceptron_delta(self, dataset_id: str, c1: str, c2: str) -> dict:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron com regra delta.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se o nome da classe não for válido
        """
        raise NotImplementedError
        
        if not dataset_id.lower() in self.datasets:
            raise InvalidDataset("Dataset not found")
        if not c1 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")
        if not c2 in self.datasets[dataset_id]["dataset"]:
            raise InvalidClass("Invalid class name")

        dataset = self.datasets[dataset_id.lower()]["dataset"]
        c1v = dataset[c1]["train"].lines
        c2v = dataset[c2]["train"].lines

        func, eq, iters = classifier.delta_perceptron(c1v, c2v)
        # TODO: Reduzir o delta quando soltar erro ou receber delta como argumento?

        results = {
            c : [{"data": sample, "result": func(sample)} for sample in dataset[c]["test"].lines]
            for c in ("setosa", "versicolor", "virginica")
        }

        return {
            "eq": eq,
            "iters": iters,
            "test_result": results
        }

    def compare_classifiers(self, dataset_id: str, alg1: str, alg2: str) -> dict:
        """
        Calcula as métricas de tau e kappa para dois classificadores.
        :param dataset_id: o id do conjunto de dados.
        :param alg1: o nome de um algoritmo de classificação.
        Deve ser "perceptron", "perceptron_delta", "euclidean", "maximum" ou "dij"
        :param alg2: o nome de um algoritmo de classificação.
        Deve ser "perceptron", "perceptron_delta", "euclidean", "maximum" ou "dij"
        :returns: Um dicionário contendo os resultados da comparação.
        """
        if alg1 not in ("perceptron", "perceptron_delta", "euclidean", "maximum", "dij"):
            raise InvalidAlgorithm
        if alg2 not in ("perceptron", "perceptron_delta", "euclidean", "maximum", "dij"):
            raise InvalidAlgorithm

        raise NotImplementedError

        return {
            
        }