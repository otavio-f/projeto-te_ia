"""
Testes de unidade
"""

import math
import unittest
from vector import Vector


class VectorTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de vetores
    """

    data_sample = [
        [1, 2, 3],
        [4, 5, 6],
        [1, 1, 1],
        [0, 0, 0]
    ]

    def testAddVectorPlusVector(self):
        """Testa adição de dois vetores elemento a elemento."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])

        result = v1 + v2
        self.assertEqual(result, [5, 7, 9])

    def testAddVectorPlusScalar(self):
        """Testa adição de vetor a escalar."""
        v = Vector([1, 2, 3])
        k = 10

        result = v + k
        self.assertDictEqual(result, [11, 12, 13])

    def testSubVectorDiffVector(self):
        """Testa subtração de dois vetores elemento a elemento."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])

        result = v1 + v2
        self.assertEqual(result, [-3, -3, -3])

    def testSubVectorDiffScalar(self):
        """Testa subtração de vetor a escalar."""
        v = Vector([1, 2, 3])
        k = 10

        result = v + k
        self.assertDictEqual(result, [-9, -8, -7])

    def testMulVectorToVector(self):
        """Testa combinação de dois vetores."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])

        result = v1 * v2

        self.assertEqual(result, 32)
        self.assertEqual(v1*v2.T, 32)
        self.assertEqual(v1.T*v2, 32)

    def testMulVectorToScalar(self):
        """Testa multiplicação de vetor por escalar."""
        v1 = Vector([1, 2, 3])
        v2 = 10

        result = v1 * v2

        self.assertEqual(result, [10, 20, 30])

    def testVectorUnaryNeg(self):
        """
        Testa negação do vetor.
        Deve se comportar do mesmo modo da multiplicação por -1 escalar.
        """
        v = Vector([1, 2, 3])
        
        result = -v
        self.assertDictEqual(result, [-1, -2, -3])
        self.assertEqual(result, v * -1)


class ClassifierTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de classificador de distância mínima
    """

    def testEuclideanDist(self):
        """
        Testa distância euclideana entre dois vetores.
        """
        cl = Classifier("teste", {"x1":math.pi, "x2":math.e, "x3":1})
        result = cl.euclidean_dist({"x1": 1, "x2":1, "x3":1}) # usando duas coordenadas das três

        self.assertAlmostEqual(result, 2.74, delta = 0.01)

    def testMaxDist(self):
        """
        Testa classificador de máxima.
        """
        c1 = Classifier("teste", {"x1":math.pi, "x2":math.e, "x3":1})
        c2 = Classifier("teste2", {"x1":0, "x2":0, "x3":0})
        c3 = Classifier("teste3", {"x1":100, "x2":10, "x3":-10})

        test = {"x1": 1, "x2":1, "x3":1}
        results = tuple(c.max_dist(test) for c in (c1, c2, c3))

        self.assertEqual(max(results), results[1]) # o segundo (c2)

    def testDIJ(self):
        """
        Testa superfície de decisão.
        """
        c1 = Classifier("teste", {"x1":math.pi, "x2":math.e, "x3":1})
        c2 = Classifier("teste2", {"x1":0, "x2":0, "x3":0})

        test = {"x1": 3, "x2":2, "x3":1}
        func = c1.dij(c2)
        result = func(test)

        self.assertTrue(result > 0) # > 0 pertence a primeira classe (c1)


if __name__ == "__main__":
    unittest.main()
