from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field

import random
import os
from typing import Callable
import numpy as np

from dataset import DataSet, DataExtractor
import classifiers as classifier


@dataclass(frozen=True)
class Sample:
    "Armazena os dados de uma amostra."
    data: np.ndarray
    "Vetor da amostra"

    cls: str
    "Classe a qual a amostra pertence"


@dataclass(frozen=True)
class ClassifiedSample:
    "Armazena os dados de uma amostra classificada."
    data: np.ndarray
    "Vetor da amostra"

    expected: str
    "Classe a qual a amostra pertence"

    value: float = None
    "Resultado da aplicação do vetor da amostra à equação de classificação."

    predicted: str = None
    "Classe atribuída a amostra"


@dataclass(frozen=True)
class TrainSet:
    "Armazena um dataset de treino."

    cls: str
    "Classe de treino."

    dataset: DataSet
    "Vetores de treino."

    length: int
    "Quantidade de vetores de treino."


@dataclass(frozen=True)
class DataSetInfo:
    """
    Armazena informações sobre o dataset.
    """

    id: str
    "Valor de identificação do dataset."

    classes: list[str]
    "Classes de treinamento."

    columns: list[str]
    "Colunas usadas no treinamento."

    train_percent: float
    "Porcentagem de dados usados no treino."

    train_set: list[TrainSet]
    "Os datasets usados no treinamento."

    test_set: list[Sample]
    "As amostras usadas para teste."

    def get_trainset(self, cls: str) -> TrainSet|None:
        """
        Retorna o dataset de treino da classe.
        :param cls: O nome da classe.
        :returns: O dataset de treino ou None se a classe não for encontrada.
        """
        return next((x for x in self.train_set if x.cls == cls),
                    None)

    def __hash__(self) -> int:
        """Calcula o hash desse objeto."""
        return hash(self.id)


    @staticmethod
    def __create_id() -> str:
        """
        Cria um novo id.
        :return: um id de 8 algarismos hexadecimais aleatórios.
        """
        new_id = random.choices("abcdef1234567890", k=8)
        return "".join(new_id)


    @staticmethod
    def from_disk(train_percent: float, columns: list[str], src: os.PathLike, classes: list[str], class_col_name: str) -> 'DataSetInfo':
        """
        Recupera um dataset a partir de um arquivo.

        :param train_percent: A porcentagem de treino a ser usada
        :param columns: As colunas a serem selecionadas
        :param src: O arquivo fonte dos dados
        :param classes: Todas as classes possíveis
        :param class_col_name: O nome da coluna que contém as classes
        :returns: Uma instância de classe contendo a informação sobre o Dataset gerado
        """
        # classes = ("setosa", "versicolor", "virginica")
        TrainTest = namedtuple("TrainTest", ("train", "test"))
        
        # extrai dados de cada classe
        extractor = DataExtractor(src)
        conversor = lambda x: float(x.replace(",", "."))
        data = {
            cls: TrainTest(train, test)
            for cls in classes
            for (train, test) in extractor.extract_by_class(class_col_name, cls, columns, conversor).get_training_data(train_percent)
            # for (train, test) in extractor.extract_by_class("Species", cls, columns, conversor).get_training_data(train_percent)
        }
        
        # mistura as amostras de teste em um único conjunto
        test_set = [Sample(v, cls)
                    for cls in classes
                    for v in data[cls].test]
        
        # constroi dados de treino
        train_sets = [TrainSet(cls, data[cls].train, len(data[cls].train)) for cls in classes]

        return DataSetInfo(
            DataSetInfo.__create_id(),
            classes,
            columns,
            train_percent,
            train_sets,
            test_set
        )


@dataclass(frozen=True)
class TrainResult:
    """Representa o resultado do teste de um algoritmo de classificação."""
    equation: str
    "representação da equação de classificação"

    function: Callable[[list[float]], float]
    "função de classificação que recebe uma amostra e retorna um valor"

    tests: list[Sample]
    "coleção de resultados dos testes"

    iterations: int = -1
    """
    número de iterações necessárias para convergência do algoritmo.
    Valores negativos significam que o algoritmo não é iterativo.
    """


@dataclass
class Controller:
    """Controlador da aplicação."""

    datasets: set[DataSetInfo] = field(default_factory=set)


    def __get_dataset(self, ds_id: str) -> DataSetInfo|None:
        """
        Retorna um dataset com o id especificado.

        :returns: Uma instância de datasetinfo None se não existir
        """
        return next(
            (x for x in self.datasets if x.id == ds_id.lower()),
            None)

    def gen_dataset(self, train_percent: float, columns: list[str], src="data/Iris.csv") -> str:
        """
        Gera um conjunto de dados.

        :param train_percent: A porcentagem de dados a serem separados para treino
        :param columns: As colunas selecionadas
        :param src: A fonte de dados, por padrão o conjunto de dados iris
        :returns: O id do conjunto de dados criado.
        """
        result = DataSetInfo.from_disk(
            train_percent,
            columns,
            src,
            ("setosa", "versicolor", "virginica"),
            "Species")
        self.datasets.add(result)
        return result.id
    
    def get_all_datasets(self) -> tuple:
        """
        Recupera informação sobre todos os conjuntos de dados.

        :returns: Os conjuntos de dados.
        """
        return tuple(self.datasets)


    def get_dataset(self, dataset_id: str) -> DataSetInfo|None:
        """
        Recupera um conjunto de dados com o id especificado.

        :returns: O conjunto de dados ou None se o dataset não existir.
        """
        return self.__get_dataset(dataset_id)


    def euclidean_distance(self, dataset_id: str) -> TrainResult:
        """
        Calcula a distância euclideana do conjunto de dados de teste e as dto.SPECIES.
        
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        raise NotImplementedError


    def euclidean_distance(self, dataset_id: str, class_name: str) -> TrainResult:
        """
        Calcula a distância euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if class_name not in dataset.classes:
            raise ValueError("Invalid class name")



    def maximum(self, dataset_id: str, class_name: str) -> dict:
        """
        Calcula a distância euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param class_name: o nome da classe
        :returns: Um dicionário com a equação e os resultados dos testes.
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if class_name not in dataset.classes:
            raise ValueError("Invalid class name")



    def dij(self, dataset_id: str, c1: str, c2: str) -> 'Equation':
        """
        Calcula a superfície de decisão euclideana do conjunto de dados de teste e a classe.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if not c1 not in dataset.classes or not c2 in dataset.classes:
            raise ValueError("Invalid class name")

        raise NotImplementedError


    def perceptron(self, dataset_id: str, c1: str, c2: str, max_iters: int) -> dict:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :param max_iters: o número máximo de iterações permitido
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if not c1 not in dataset.classes or not c2 in dataset.classes:
            raise ValueError("Invalid class name")

        raise NotImplementedError


    def perceptron_delta(self, dataset_id: str, c1: str, c2: str, max_iters: int) -> dict:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron com regra delta.
        :param dataset_id: o id do conjunto de dados
        :param c1: o nome da primeira classe
        :param c2: o nome da segunda classe
        :param max_iters: o número máximo de iterações permitido
        :returns: Um dicionário com a equação e os resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """

        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if not c1 not in dataset.classes or not c2 in dataset.classes:
            raise ValueError("Invalid class name")

        raise NotImplementedError


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
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise ValueError("Dataset not found")

        if alg1 not in ("perceptron", "perceptron_delta", "euclidean", "maximum", "dij"):
            raise ValueError
        if alg2 not in ("perceptron", "perceptron_delta", "euclidean", "maximum", "dij"):
            raise ValueError

        raise NotImplementedError
