"""
Classificador de distância mínima
:author: Otávio Ferreira
"""

import math
from dataclasses import dataclass
from dataclasses import field
from vector.vector import Vector


@dataclass
class Classifier:
    """
    Classe classificadora de distância mínima
    """
    name: str
    m: Vector

    @property
    def m(self) -> Vector:
        return self._m

    @m.setter
    def m(self, value: list):
        """Atribui o valor da propriedade."""
        self._m = Vector(value)

    def euclidean_dist(self, sample: list) -> float:
        """
        Calcula a distância euclideana entre a amostra e o Classificador
        :param sample: Um vetor de amostras
        :return: A distância euclideana
        :raise ValueError: Se a amostra não é compatível com o classificador
        """
        x = Vector(sample)
        mj = self.m
        return math.sqrt((x - mj).T * (x - mj))

    def max_dist(self, sample: list) -> float:
        """
        Calcula a menor distância entre os classificadores e a amostra
        :param sample: Um vetor de amostras
        :return: O classificador com a menor distância
        :raise ValueError: Se a amostra não é compatível com o classficador
        """
        x = Vector(sample)
        mj = self.m
        return (x.T * mj) - (1/2 * (mj.T * mj))

    def dij(self, classifier: 'Classifier') -> 'lambda':
        """
        Calcula a superfície de decisão
        :param classifier: o outro classificador
        :return: Uma função de regra de decisão
        """
        mi = self.m
        mj = classifier.m
        return lambda x: ((mi - mj).T * x) - 1/2 * ((mi - mj).T * (mi + mj))
