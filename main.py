"""
Reconhecimento com supervisão
"""

"""
Classificador de distância mínima
"""
import math
import csv
import random
from dataclasses import dataclass


class Vector(dict):
    """
    Define um vetor com índices nomeados
    """
    def __mul__(self, another: dict|float|int) -> 'float | Vect':
        """
        Multiplica esse vetor a um número ou outro vetor
        :param another: Um vetor ou escalar a ser multiplicado
        :returns: Um novo vetor
        """
        if isinstance(another, dict):
            if all(k in another.keys() for k in self.keys()):
                return sum(self[k] * another[k] for k in self.keys())
            if all(k in self.keys() for k in another.keys()):
                return sum(self[k] * another[k] for k in another.keys())
            raise ValueError("Vetores Incompatíveis")
        return Vector((k, v*another) for k, v in self.items())

    def __add__(self, another: dict|float|int) -> 'Vect':
        """
        Soma esse vetor a um número ou outro vetor
        :param another: Um vetor ou escalar a ser somado
        :returns: Um novo vetor
        """
        if isinstance(another, dict):
            if all(k in another.keys() for k in self.keys()):
                return Vector({k: self[k] + another[k] for k in self.keys()})
            if all(k in self.keys() for k in another.keys()):
                return Vector({k: self[k] + another[k] for k in another.keys()})
            raise ValueError("Vetores Incompatíveis")
        return Vector((k, v+another) for k, v in self.items())

    def __sub__(self, another: dict|float|int) -> 'Vect':
        """
        Subtrai um número ou outro vetor desse vetor
        :param another: Um vetor ou escalar a ser subtraído
        :returns: Um novo vetor
        """
        if isinstance(another, dict):
            if all(k in another.keys() for k in self.keys()):
                return Vector({k: self[k] - another[k] for k in self.keys()})
            if all(k in self.keys() for k in another.keys()):
                return Vector({k: self[k] - another[k] for k in another.keys()})
            raise ValueError("Vetores Incompatíveis")
        return Vector((k, v-another) for k, v in self.items())

    def pow2(self) -> float:
        """
        Eleva esse vetor ao quadrado
        :returns: O resultado da multiplicação
        """
        return self * self

    def __neg__(self) -> 'Vect':
        """
        Inverte o sinal dos elementos do array
        :returns: Um novo vetor
        """
        return Vector((k, -v) for k,v in self.items())


@dataclass
class Classifier:
    """
    Classe classificadora
    """
    name: str
    m: Vector


    @property
    def m(self) -> Vector:
        return self._m

    @m.setter
    def m(self, value: dict):
        """Atribui o valor da propriedade."""
        self._m = Vector(value)

    def euclidean_dist(self, sample: dict) -> float:
        """
        Calcula a distância euclideana entre a amostra e o Classificador
        :param sample: Um vetor de amostras
        :return: A distância euclideana
        :raise ValueError: Se a amostra não é compatível com o classificador
        """
        x = Vector(sample)
        mj = self.m
        return math.sqrt((x - mj) * (x - mj))

    def max_dist(self, sample: dict) -> float:
        """
        Calcula a menor distância entre os classificadores e a amostra
        :param sample: Um vetor de amostras
        :return: O classificador com a menor distância
        :raise ValueError: Se a amostra não é compatível com o classficador
        """
        x = Vector(sample)
        mj = self.m
        return (x*mj) - (1/2 * mj.pow2())

    def dij(self, classifier: 'Classifier') -> 'lambda':
        """
        Calcula a superfície de decisão
        :param classifier: o outro classificador
        :return: Uma função de regra de decisão
        """
        mi = self.m
        mj = classifier.m
        return lambda x: ((mi - mj) * x) - 1/2*((mi - mj) * (mi + mj))

    def train(src: 'IO', class_name: str, spec_col='Species', sample_size=0.7) -> 'Classifier':
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


if __name__ == "__main__":
    with open('data/Iris data.csv', mode='r', newline='') as iris_data:
        # cria classificadores
        setosa = Classifier.train(iris_data, "setosa")
        versicolor = Classifier.train(iris_data, "versicolor")
        virginica = Classifier.train(iris_data, "virginica")

        print("Classificadores: ")
        print(setosa)
        print(versicolor)
        print(virginica)

        # cria amostras de teste
        sample_setosa = {'Sepal length': 5.1, 'Sepal width': 3.5, 'Petal length': 1.4, 'Petal width': 0.2}
        sample_versicolor = {'Sepal length': 7, 'Sepal width': 3.2, 'Petal length': 4.7, 'Petal width': 1.4}
        sample_virginica = {'Sepal length': 6.3, 'Sepal width': 2.7, 'Petal length': 4.9, 'Petal width': 1.8}
        
        # teste
        print(f"\nDistancia classificador setosa x amostra setosa: {setosa.euclidean_dist(sample_setosa):.2f}")
        print(f"Distancia classificador setosa x amostra versicolor: {setosa.euclidean_dist(sample_versicolor):.2f}")
        print(f"Distancia classificador setosa x amostra virginica: {setosa.euclidean_dist(sample_virginica):.2f}")

        print(f"\nDistancia classificador versicolor x amostra setosa: {versicolor.euclidean_dist(sample_setosa):.2f}")
        print(f"Distancia classificador versicolor x amostra versicolor: {versicolor.euclidean_dist(sample_versicolor):.2f}")
        print(f"Distancia classificador versicolor x amostra virginica: {versicolor.euclidean_dist(sample_virginica):.2f}")

        print(f"\nDistancia classificador virginica x amostra setosa: {virginica.euclidean_dist(sample_setosa):.2f}")
        print(f"Distancia classificador virginica x amostra versicolor: {virginica.euclidean_dist(sample_versicolor):.2f}")
        print(f"Distancia classificador virginica x amostra virginica: {virginica.euclidean_dist(sample_virginica):.2f}")

