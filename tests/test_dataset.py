"""
Testes de unidade para módulo conjunto de dados
"""

import math
import unittest
from vector.vector import Vector
from vector.dataset import DataSet


class DatasetTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de conjunto de dados
    """
    def testMean(self):
        """Testa cálculo da média do vetor."""
        vs = [Vector([1, 2, 3]), Vector([0, 0, 0])]
        d1 = DataSet(vs)

        result = d1.m
        self.assertEqual(result, (2.0, 0.0))

    """
    Classe de teste de unidade de conjunto de dados
    """
    def testGetTrainingData(self):
        """Testa separação de dados em treino e teste."""
        vs = [
            Vector([1, 2, 3, 4]),
            Vector([0, 0, 0, 0]),
            Vector([1, 10, 100, 1000])
            ]
        d1 = DataSet(vs)

        train, test = d1.get_training_data(0.5)

        self.assertEqual(len(train.columns), 3)
        self.assertEqual(len(test.columns), 3)

        self.assertEqual(len(train.columns[0]), 2)
        self.assertEqual(len(test.columns[0]), 2)

        self.assertEqual(sorted(list(train[0])+list(test[0])), [1, 2, 3, 4])
        self.assertEqual(sorted(list(train[1])+list(test[1])), [0, 0, 0, 0])
        self.assertEqual(sorted(list(train[2])+list(test[2])), [1, 10, 100, 1000])
