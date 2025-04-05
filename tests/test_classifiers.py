import unittest
import math
import classifier.classifiers as classifier
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
        func, equation = classifier.euclidean_dist(test)

        expected_eq = "√(x1²+x2²+x3²-8x1-10x2-12x3+77)"
        self.assertEqual(expected_eq, equation)

        self.assertAlmostEqual(math.sqrt(77), func((0, 0, 0)), delta=0.001)

        self.assertAlmostEqual(0, func((4, 5, 6)), delta=0.001)
    
    def testMaxDist(self):
        """
        Testa classificador máximo.
        """
        test = np.array((3, 4, 5))
        func, equation = classifier.max_dist(test)

        expected_eq = "3x1+4x2+5x3-25.0"
        self.assertEqual(expected_eq, equation)

        self.assertAlmostEqual(-25, func((0, 0, 0)), delta=0.001)

        self.assertAlmostEqual(25, func((3, 4, 5)), delta=0.001)
    
    def testDIJ(self):
        """
        Testa superfície de decisão.
        """
        versicolor = np.array((4.3, 1.3))
        setosa = np.array((1.5, 0.3))

        func, equation = classifier.dij(versicolor, setosa)

        expected_eq = "2.8x1+x2-8.92"
        self.assertEqual(expected_eq, equation)

        sample_setosa = (1.4, 0.2)
        self.assertTrue(func(sample_setosa) < 0)
    
    def testPerceptron(self):
        """
        Testa geração da superfície de decisão pelo algoritmo Perceptron.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        func, eq, iters = classifier.perceptron(cl1, cl2)

        self.assertEqual("-2x1+1", eq)
        self.assertEqual(16, iters)
        self.assertTrue(func([0, 0]) > 0)
        self.assertTrue(func([0, 1]) > 0)
        self.assertTrue(func([1, 0]) < 0)
        self.assertTrue(func([1, 1]) < 0)
    
    def testDeltaPerceptron(self):
        """
        Testa perceptron com regra delta.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        func, eq, iters = classifier.delta_perceptron(cl1, cl2, alpha=0.1)

        # self.assertEqual("", eq)
        self.assertTrue(func([0, 0]) > 0)
        self.assertTrue(func([0, 1]) > 0)
        self.assertTrue(func([1, 0]) < 0)
        self.assertTrue(func([1, 1]) < 0)
    