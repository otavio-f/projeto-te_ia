from dataclasses import dataclass, field
import csv
import random
import os
import math
from vector.vector import Vector


@dataclass
class DataSet:
    """
    Representa um conjunto de dados
    """
    __columns: [Vector, ...]
    __len: int = field(init=False)

    def __post_init__(self):
        """
        Verifica a consistência dos dados de entrada.
        """
        size = len(self.__columns[0])
        if not all(len(col) == size for col in self.__columns):
            raise ValueError("Mismatching columns!")
        self.__len = size

    def __getitem__(self, i: 'str|int'):
        """
        Obtém um item do conjunto através do nome ou do índice.
        :param i: Índice ou nome do vetor
        :returns: O vetor ou ValueError se não houver vetor com nome ou índice válido
        """
        try:
            return self.__columns[i]
        except TypeError:
            for v in self.__columns:
                if v.title == i:
                    return v
        raise ValueError("Key not found!")

    @property
    def columns(self) -> [Vector, ...]:
        """
        Obtém as colunas desse conjunto de dados.
        """
        return self.__columns

    @property
    def m(self) -> [float, ...]:
        """
        Calcula a média de todos os vetores.
        :returns: A média de cada vetor desse conjunto de dados.
        """
        return tuple(float(v.m) for v in self.__columns)

    def get_training_data(self, percentage: float) -> '[DataSet, DataSet]':
        """
        Separa o conjunto de dados para treino e teste.
        :param percentage: A porcentagem a ser destinada a testes.
        :returns: Dois conjuntos de dados, um para treino e outro para testes.
        """
        # transform to list
        all_data = (list(v) for v in self.__columns)
        names = [v.title for v in self.__columns]
        # bind columns together
        all_data = list(zip(*all_data))
        # shuffle
        random.shuffle(all_data)
        # determine the cut point between training and testing
        cut_point = math.floor(self.__len * percentage)
        # separe into train and test
        train_data = all_data[:cut_point]
        test_data = all_data[cut_point:]
        # unbind data
        # dica: zip() todas as colunas juntas
        # dica: o inverso de zip() é zip()
        train_data = tuple(zip(*train_data))
        test_data = tuple(zip(*test_data))
        # Re-vector columns
        train_data = [Vector(data, name) for name,data in zip(names, train_data)]
        test_data = [Vector(data, name) for name,data in zip(names, test_data)]
        return DataSet(train_data), DataSet(test_data)

@dataclass(frozen=True)
class DataExtractor:
    src: os.PathLike

    def extract_by_class(self, class_column: str, class_name: str, data_columns: [str], cf: 'Callable[[str], float]'=None) -> DataSet:
        with open(self.src, mode='r', newline='') as fp:
            reader = csv.reader(fp)
            # get headers
            headers = next(reader)
            # extract column indexes from headers
            cols_i = zip(tuple(headers.index(col_name) for col_name in data_columns), data_columns)
            class_i = headers.index(class_column)
            # select rows of the wanted class
            all_rows = tuple(row for row in reader if row[class_i] == class_name)
            # filter and optionally convert the wanted columns
            if cf is None:
                columns = tuple(Vector(tuple(row[i] for row in all_rows), name) for i, name in cols_i)
            else:
                columns = tuple(Vector(tuple(cf(row[i]) for row in all_rows), name) for i, name in cols_i)
            return DataSet(columns)

    def train(src: 'IO', class_name: str, spec_col='Species', sample_size=0.7) -> 'Classifier':
        # TODO: Extrair captura e parsing e separação de dados treino/teste para outro módulo.
        # Aqui só fica treinamento
        """
        Treina a base de dados.
        :param src: Fonte de dados
        :param class_name: nome da classe a ser treinada
        :param spec_col: nome da coluna que designa a espécie
        :param sample_size: tamanho da amostra para treino
        :return: Um classificador
        """
        src.seek(0) # reseta leitor para a primeira linha
        reader = csv.DictReader(src)
        rows = tuple(row for row in reader if row[spec_col] == class_name) # separa a classe
        data_size = math.floor(len(rows) * sample_size) # calcula a quantidade de dados de treino
        sample = random.sample(rows, data_size) # obtém amostra de treino
        ms = Vector() # inicializa vetor de médias
        for col in reader.fieldnames: # calcula médias
            if(col == spec_col):
                continue
            median = sum(float(row[col].replace(',', '.')) for row in sample)/len(sample)
            ms[col] = median
        return Classifier(class_name, ms)

if __name__ == '__main__':
    ext = DataExtractor('data/Iris data.csv')
    convert = lambda s: float(s.replace(',', '.'))
    columns = ["Sepal length","Sepal width"]
    dt = ext.extract_by_class("Species", "setosa", columns, convert)
    dt.get_training_data(0.8)