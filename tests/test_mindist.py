import unittest
import math
from classifier.mindist import Classifier


class ClassifierTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de classificador de distância mínima
    """

    def testEuclideanDist(self):
        """
        Testa distância euclideana entre dois vetores.
        """
        cl = Classifier("teste", (math.pi, math.e, 1))
        result = cl.euclidean_dist((1, 1, 1)) # usando duas coordenadas das três

        self.assertAlmostEqual(result, 2.74, delta = 0.01)

    def testMaxDist(self):
        """
        Testa classificador de máxima.
        """
        c1 = Classifier("teste", (math.pi, math.e, 1))
        c2 = Classifier("teste", (0, 0, 0))
        c3 = Classifier("teste", (100, 10, -10))
        
        test = (1, 1, 1)
        results = tuple(c.max_dist(test) for c in (c1, c2, c3))

        self.assertEqual(max(results), results[1]) # o segundo (c2)

    def testDIJ(self):
        """
        Testa superfície de decisão.
        """
        c1 = Classifier("teste", (math.pi, math.e, 1))
        c2 = Classifier("teste", (0, 0, 0))

        test = (3, 2, 1)
        func = c1.dij(c2)
        result = func(test)

        self.assertTrue(result > 0) # > 0 pertence a primeira classe (c1)
