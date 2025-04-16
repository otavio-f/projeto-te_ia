from typing import Tuple
from typing import Any
from typing import Callable

from dataclasses import dataclass, field
import csv
import random
import os
import math
import numpy as np


@dataclass(frozen=True)
class DataSet:
    """
    Representa um conjunto de dados
    """
    __columns: Tuple[np.ndarray, ...]
    column_titles: Tuple[str, ...]

    def __post_init__(self):
        """
        Verifica a consistência dos dados de entrada.
        """
        
        # quantidade de colunas x quantidade de titulos
        assert len(self.__columns) == len(self.column_titles)
        
        # todas colunas devem ter mesma quantidade de elementos
        length = self.__columns[0].shape
        assert all(col.shape == length for col in self.__columns), "Mismatched columns!"

    def __getitem__(self, i: 'str|int'):
        """
        Obtém um item do conjunto através do nome ou do índice.
        :param i: Índice ou nome do vetor
        :returns: O vetor ou ValueError se não houver vetor com nome ou índice válido
        """
        if type(i) is int:
            return self.__columns[i]
        elif type(i) is str:
            result = filter(lambda col: col.title==i, self.__columns)
            try:
                return next(result)
            except StopIteration: # não existe resultado
                return None
        else:
            raise ValueError("Argumento inválido!")

    @property
    def columns(self) -> np.ndarray:
        """
        Obtém as colunas desse conjunto de dados.
        """
        # retorna c colunas com n elementos cada
        # [
        # [ col[0][0], col[0][1], ..., col[0][n] ],
        # [ col[1][0], col[1][1], ..., col[1][n] ],
        # ...,
        # [ col[c][0], col[c][1], ..., col[c][n] ]
        # ]
        return np.stack(self.__columns)

    @property
    def lines(self) -> np.ndarray:
        """
        Obtém as linhas desse conjunto de dados.
        """
        # retorna o n elemento de cada coluna em c colunas
        # [
        # [ col[0][0], col[1][0], ..., col[c][0] ],
        # [ col[0][1], col[1][1], ..., col[c][1] ],
        # ...,
        # [ col[0][n], col[1][n], ..., col[c][n] ]
        # ]
        return np.column_stack(self.__columns)

    @property
    def m(self) -> np.ndarray:
        """
        Calcula a média de todas as colunas.
        :returns: A média de cada vetor desse conjunto de dados.
        """
        return np.fromiter((col.mean() for col in self.__columns), np.single)

    def get_training_data(self, percentage: float) -> Tuple['DataSet', 'DataSet']:
        """
        Separa o conjunto de dados para treino e teste.
        :param percentage: A porcentagem a ser destinada a testes.
        :returns: Dois conjuntos de dados, um para treino e outro para testes.
        """
        # determina ponto de corte entre treinamento e teste
        cut_point = math.floor(len(self.__columns[0]) * percentage)

        # mistura linhas
        lines = self.lines
        rng = np.random.default_rng()
        rng.shuffle(lines)
        
        # separa treino e teste
        train = lines[:cut_point] # treino: de 0 ao ponto de corte
        test = lines[cut_point:] # teste: do ponto de corte ao final
        
        # transforma linhas em colunas
        trainset = DataSet(tuple(train.T), self.column_titles)
        testset = DataSet(tuple(test.T), self.column_titles)

        # retorna resultados
        return trainset, testset


@dataclass(frozen=True)
class DataExtractor:
    src: os.PathLike

    def extract_by_class(self, class_column: str, class_name: str, data_columns: Tuple[str, ...], conversor: Callable[[str], Any]=None) -> DataSet:
        with open(self.src, mode='r', newline='') as fp:
            reader = csv.reader(fp)

            # cabeçalhos
            headers = next(reader)
            
            # extrai índices dos cabeçalhos
            columns_index = tuple(headers.index(name) for name in data_columns) # colunas selecionadas
            class_index = headers.index(class_column) # coluna da classe

            # filtra linhas e colunas
            all_rows = tuple([] for _ in columns_index)
            for row in reader:
                # descarta linhas das classes erradas
                if row[class_index] != class_name:
                    continue
                # adiciona dados às colunas selecionadas pelo argumento
                for i, ci in enumerate(columns_index):
                    all_rows[i].append(row[ci]) # BUG: all_rows tem menos colunas que columns_index

            # filtra e opcionalmente converte as colunas obtidas
            if conversor is None: # não tem conversor
                data = tuple(np.array(row, np.single)
                    for row in all_rows)
            else:
                data = tuple(np.fromiter((conversor(i) for i in row), np.single)
                    for row in all_rows)
            
            return DataSet(data, data_columns)


if __name__ == '__main__':
    ext = DataExtractor('data/Iris.csv')
    conversor = lambda s: s.replace(',', '.')
    columns = ["Sepal length", "Sepal width"]
    dt = ext.extract_by_class("Species", "setosa", columns, conversor)
    train, test = dt.get_training_data(0.8)

    print(train.m)