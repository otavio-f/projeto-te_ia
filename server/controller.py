from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field

import itertools
import random
import os
from typing import NamedTuple
import numpy as np

from processors.dataset import DataSet, DataExtractor

from processors.classifiers import Classifier
from processors.evaluators import ConfusionMatrix
from processors.evaluators import BinaryMatrix


class Sample(NamedTuple):
    "Armazena os dados de uma amostra."
    data: np.ndarray
    "Vetor da amostra"

    cls: str
    "Classe a qual a amostra pertence"

    def classify(self, value: float, predicted: str) -> 'ClassifiedSample':
        """
        Classifica uma amostra.
        
        :param value: O resultado da equação da classificação.
        :param predicted: A classe atribuída pela classificação.
        """
        return ClassifiedSample(self.data, self.cls, value, predicted)


class ClassifiedSample(NamedTuple):
    "Armazena os dados de uma amostra classificada."
    data: np.ndarray
    "Vetor da amostra"

    expected: str
    "Classe a qual a amostra pertence"

    value: float = None
    "Resultado da aplicação do vetor da amostra à equação de classificação."

    predicted: str = None
    "Classe atribuída a amostra"


class DataSetInfo(NamedTuple):
    """
    Armazena informações sobre o dataset.
    """

    id: str
    "Valor de identificação do dataset."

    classes: tuple[str]
    "Classes de treinamento."

    columns: tuple[str]
    "Colunas usadas no treinamento."

    train_percent: float
    "Porcentagem de dados usados no treino."

    train_set: dict[str, DataSet]
    "Os datasets usados no treinamento."

    test_set: tuple[Sample]
    "As amostras usadas para teste."

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
        data: dict[str, TrainTest] = dict()

        for cls in classes:
            train, test = extractor.extract_by_class(class_col_name, cls, columns, conversor).get_training_data(train_percent)
            data[cls] = TrainTest(train, test)
        
        # mistura as amostras de teste em um único conjunto
        test_set = tuple(Sample(v, cls)
                    for cls in classes
                    for v in data[cls].test.lines)
        
        # constroi dados de treino
        train_sets: dict[str, DataSet] = dict()
        for cls in classes:
            train_sets[cls] = data[cls].train
        # train_sets = [TrainSet(cls, data[cls].train, len(data[cls].train.lines)) for cls in classes]

        return DataSetInfo(
            DataSetInfo.__create_id(),
            tuple(classes),
            tuple(columns),
            train_percent,
            train_sets,
            test_set
        )


class TrainResult(NamedTuple):
    """Representa o resultado do teste de um algoritmo de classificação."""

    classifier: dict[str, Classifier]
    "O algoritmo de classificação treinado."
    
    tests: list[ClassifiedSample]
    "coleção de resultados dos testes"

    matrix: ConfusionMatrix
    "Matriz de confusão"

    bin_matrixes: dict[str, BinaryMatrix]
    "Matrizes binárias de cada classe"

    @staticmethod
    def create(classes: list[str], classifier: dict[str, Classifier], tests: list[ClassifiedSample]):
        """
        Compila dados em um resumo de treino.
        
        :param classes: As classes usadas no treino e testes.
        :param classifiers: Os classificadores para cada classe.
        :param tests: Os resultados dos testes.
        
        :returns: Um objeto resumo do treino e testes.
        """
        # cria matriz CxC, C sendo a quantidade de classes
        m = [
            [0 for _ in classes]
            for _ in classes ]
        
        # atribui valores da matriz de confusão
        for sample in tests:
            expectIndex = classes.index(sample.expected) # seleciona coluna
            predictIndex = classes.index(sample.predicted) # seleciona linha
            m[expectIndex][predictIndex] += 1

        # gera matriz de confusão
        matrix = ConfusionMatrix(m, classes)

        # gera matrizes binárias
        bins = {
            cls: BinaryMatrix.from_confusion_matrix(matrix, cls)
            for cls in classes
            }

        return TrainResult(classifier, tests, matrix, bins)
        

class Controller(NamedTuple):
    """Controlador da aplicação."""

    datasets: set[DataSetInfo]

    @staticmethod
    def create() -> 'Controller':
        """
        Cria uma instância de controlador.
        
        :returns: Um controlador padrão.
        """
        return Controller(set())


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
    
    def get_all_datasets(self) -> tuple[DataSetInfo]:
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
        Calcula a distância euclideana do conjunto de dados de teste e as classes.
        
        :param dataset_id: o id do conjunto de dados
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")

        # calcula as equações de distância euclideana de cada classe
        classifiers = dict()
        for cls in dataset.classes:
            m = dataset.train_set[cls].m
            classifiers[cls] = Classifier.euclidean_dist(m)
        
        # classifica os vetores de teste
        tests = []
        for sample in dataset.test_set:
            distances = {
                cls: classifiers[cls].func(sample.data)
                for cls in dataset.classes
                }
            mindist = min(distances.values()) # valor de distância mínima
            predict = min(dataset.classes, key=distances.get) # classe da distância mínima
            csample = sample.classify(mindist, predict)
            tests.append(csample)

        # resultado final
        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result
    

    def maximum(self, dataset_id: str) -> dict:
        """
        Calcula a máxima do conjunto de dados de teste e a classe.

        :param dataset_id: o id do conjunto de dados
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises KeyError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")
        
        # calcula as equações de distância euclideana
        classifiers = dict()
        for cls in dataset.classes:
            m = dataset.train_set[cls].m
            classifiers[cls] = Classifier.max_dist(m)
        
        # armazena os vetores de teste classificados
        tests = []
        for sample in dataset.test_set:
            distances = {
                cls: classifiers[cls].func(sample.data)
                for cls in dataset.classes
                }
            maxval = max(distances.values()) # valor de distância mínima
            predict = max(dataset.classes, key=distances.get) # classe da distância mínima
            # resultado da amostra
            csample = sample.classify(maxval, predict)
            tests.append(csample)

        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result


    def dij(self, dataset_id: str) -> TrainResult:
        """
        Calcula a superfície de decisão do conjunto de dados de teste e as classes.
        
        :param dataset_id: o id do conjunto de dados
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")
        
        # combinações de classes 2 a 2
        combinations = tuple(itertools.combinations(dataset.classes, 2))

        # calcula as superfícies de decisão de cada par de classes
        classifiers = dict()
        for c1, c2 in combinations:
            mi = dataset.train_set[c1].m
            mj = dataset.train_set[c2].m
            classifiers[(c1, c2)] = Classifier.dij(mi, mj)
        
        # armazena os vetores de teste classificados
        # TODO: generalizar para mais de três classes
        tests = []
        for sample in dataset.test_set:
            # primeira classificação usa o primeiro classificador encontrado
            c1, c2 = combinations[0]
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # escolhe o próximo par de classes do classificador
            c1, c2 = next(pair for pair in combinations # par de classes
                          if predict in pair            # par contém a classe prevista
                          and pair != (c1, c2) and pair != (c2, c1)) # par não foi usado
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # resultado da amostra
            csample = ClassifiedSample(sample.data, sample.cls,
                                           value, predict)
            tests.append(csample)

        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result


    def bayes(self, dataset_id: str) -> TrainResult:
        """
        Calcula a superfície de decisão do conjunto de dados usando o classificador de Bayes para as classes.

        :param dataset_id: o id do conjunto de dados
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")
        
        # combinações de classes 2 a 2
        combinations = tuple(itertools.combinations(dataset.classes, 2))

        # calcula as superfícies de decisão de cada par de classes
        classifiers = dict()
        for c1, c2 in combinations:
            ci = dataset.train_set[c1]
            cj = dataset.train_set[c2]
            classifiers[(c1, c2)] = Classifier.bayes(
                ci.lines.tolist(),
                cj.lines.tolist(),
                ci.m,
                cj.m)
        
        # armazena os vetores de teste classificados
        # TODO: generalizar para mais de três classes
        tests = []
        for sample in dataset.test_set:
            # primeira classificação usa o primeiro classificador encontrado
            c1, c2 = combinations[0]
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # escolhe o próximo par de classes do classificador
            c1, c2 = next(pair for pair in combinations # par de classes
                          if predict in pair            # par contém a classe prevista
                          and pair != (c1, c2) and pair != (c2, c1)) # par não foi usado
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # resultado da amostra
            csample = ClassifiedSample(sample.data, sample.cls,
                                           value, predict)
            tests.append(csample)

        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result
    
    def perceptron(self, dataset_id: str, max_iters: int) -> TrainResult:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron.

        :param dataset_id: o id do conjunto de dados
        :param max_iters: o número máximo de iterações permitido
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")

        # combinações de classes 2 a 2
        combinations = tuple(itertools.combinations(dataset.classes, 2))

        # calcula as superfícies de decisão de cada par de classes
        classifiers = dict()
        for c1, c2 in combinations:
            c1v = dataset.train_set[c1].lines
            c2v = dataset.train_set[c2].lines
            classifiers[(c1, c2)] = Classifier.perceptron(c1v, c2v, max_iters)
        
        # armazena os vetores de teste classificados
        # TODO: generalizar para mais de três classes
        tests = []
        for sample in dataset.test_set:
            # primeira classificação usa o primeiro classificador encontrado
            c1, c2 = combinations[0]
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # escolhe o próximo par de classes do classificador
            c1, c2 = next(pair for pair in combinations # par de classes
                          if predict in pair            # par contém a classe prevista
                          and pair != (c1, c2) and pair != (c2, c1)) # par não foi usado
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # resultado da amostra
            csample = ClassifiedSample(sample.data, sample.cls,
                                           value, predict)
            tests.append(csample)

        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result


    def perceptron_delta(self, dataset_id: str, max_iters: int) -> TrainResult:
        """
        Calcula a superfície de decisão usando o algoritmo perceptron com regra delta.

        :param dataset_id: o id do conjunto de dados
        :param max_iters: o número máximo de iterações permitido
        :returns: Informação sobre o algoritmo e resultados dos testes
        :raises ValueError: se algum dos parâmetros for inválido
        """
        dataset = self.__get_dataset(dataset_id)
        
        if dataset is None:
            raise KeyError("Dataset not found")

        # combinações de classes 2 a 2
        combinations = tuple(itertools.combinations(dataset.classes, 2))

        # calcula as superfícies de decisão de cada par de classes
        classifiers = dict()
        for c1, c2 in combinations:
            c1v = dataset.train_set[c1].lines
            c2v = dataset.train_set[c2].lines
            classifiers[(c1, c2)] = Classifier.delta_perceptron(c1v, c2v, max_iters)
        
        # armazena os vetores de teste classificados
        # TODO: generalizar para mais de três classes
        tests = []
        for sample in dataset.test_set:
            # primeira classificação usa o primeiro classificador encontrado
            c1, c2 = combinations[0]
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # escolhe o próximo par de classes do classificador
            c1, c2 = next(pair for pair in combinations # par de classes
                          if predict in pair            # par contém a classe prevista
                          and pair != (c1, c2) and pair != (c2, c1)) # par não foi usado
            value = classifiers[(c1, c2)].func(sample.data)
            predict = c1 if value > 0 else c2
            # resultado da amostra
            csample = ClassifiedSample(sample.data, sample.cls,
                                           value, predict)
            tests.append(csample)

        result = TrainResult.create(dataset.classes, classifiers, tests)
        return result


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
