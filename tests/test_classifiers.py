import unittest
import math
from classifiers import Classifiers as classifier
import numpy as np


class ClassifierTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de classificadores
    """

    def testEuclideanDist(self):
        """
        Testa distância euclideana.
        """

        test = np.array((4, 5, 6))
        result = classifier.euclidean_dist(test)

        expected_eq = "√(x1²+x2²+x3²-8x1-10x2-12x3+77)"
        self.assertEqual(expected_eq, result.eq)

        self.assertAlmostEqual(math.sqrt(77), result.func((0, 0, 0)), delta=0.001)

        self.assertAlmostEqual(0, result.func((4, 5, 6)), delta=0.001)
    
    def testMaxDist(self):
        """
        Testa classificador máximo.
        """
        test = np.array((3, 4, 5))
        result = classifier.max_dist(test)

        expected_eq = "3x1+4x2+5x3-25.0"
        self.assertEqual(expected_eq, result.eq)

        self.assertAlmostEqual(-25, result.func((0, 0, 0)), delta=0.001)

        self.assertAlmostEqual(25, result.func((3, 4, 5)), delta=0.001)
    
    def testDIJ(self):
        """
        Testa superfície de decisão.
        """
        versicolor = np.array((4.3, 1.3))
        setosa = np.array((1.5, 0.3))

        result = classifier.dij(versicolor, setosa)

        expected_eq = "2.8x1+x2-8.92"
        self.assertEqual(expected_eq, result.eq)

        sample_setosa = (1.4, 0.2)
        self.assertTrue(result.func(sample_setosa) < 0)
    
    def testPerceptron(self):
        """
        Testa geração da superfície de decisão pelo algoritmo Perceptron.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        result = classifier.perceptron(cl1, cl2)

        self.assertEqual("-2x1+1", result.eq)
        self.assertEqual(16, result.iters)
        self.assertTrue(result.func([0, 0]) > 0)
        self.assertTrue(result.func([0, 1]) > 0)
        self.assertTrue(result.func([1, 0]) < 0)
        self.assertTrue(result.func([1, 1]) < 0)
    
    def testDeltaPerceptron(self):
        """
        Testa perceptron com regra delta.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        result = classifier.delta_perceptron(cl1, cl2, alpha=0.1)

        # self.assertEqual("", eq)
        self.assertTrue(result.func([0, 0]) > 0)
        self.assertTrue(result.func([0, 1]) > 0)
        self.assertTrue(result.func([1, 0]) < 0)
        self.assertTrue(result.func([1, 1]) < 0)
    