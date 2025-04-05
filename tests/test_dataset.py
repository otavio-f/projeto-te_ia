"""
Testes de unidade para módulo conjunto de dados
"""

import math
import unittest
from vector.dataset import DataSet
import numpy as np

class DatasetTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de conjunto de dados
    """
    def testMean(self):
        """Testa cálculo da média do vetor."""
        vs = [np.array([1, 2, 3]), np.array([0, 0, 0])]
        d1 = DataSet(vs, ("a", "b"))

        result = d1.m
        self.assertTrue(np.array_equal(result, (2.0, 0.0)))

    """
    Classe de teste de unidade de conjunto de dados
    """
    def testGetTrainingData(self):
        """Testa separação de dados em treino e teste."""
        vs = [
            np.array([1, 2, 3, 4]),
            np.array([0, 0, 0, 0]),
            np.array([1, 10, 100, 1000])
            ]
        d1 = DataSet(vs, ("a", "b", "c"))

        train, test = d1.get_training_data(0.5)

        self.assertEqual(len(train.columns), 3)
        self.assertEqual(len(test.columns), 3)

        self.assertEqual(len(train.columns[0]), 2)
        self.assertEqual(len(test.columns[0]), 2)

        self.assertEqual(sorted(list(train[0])+list(test[0])), [1, 2, 3, 4])
        self.assertEqual(sorted(list(train[1])+list(test[1])), [0, 0, 0, 0])
        self.assertEqual(sorted(list(train[2])+list(test[2])), [1, 10, 100, 1000])
