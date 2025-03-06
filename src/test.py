"""
Testes de unidade
"""

import math
import unittest
from main import Vector, Classifier


class VectorTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de vetores
    """

    def testAddVectorPlusVector(self):
        """Testa adição de dois vetores elemento a elemento."""
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"a":3, "b":2, "c":1})

        result = v1 + v2
        self.assertDictEqual(result, {"a":4, "b":4, "c":4})

    def testAddVectorPlusScalar(self):
        """Testa adição de vetor a escalar."""
        v = Vector({"a":1, "b":2, "c":3})
        k = 10

        result = v + k
        self.assertDictEqual(result, {"a":11, "b":12, "c":13})

    def testAddVectorInvalid(self):
        """
        Testa adição de dois vetores incompatíveis.
        Deve resultar em erro.
        """
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"c":3, "d":2, "e":1})

        try:
            result = v1 + v2
            self.fail()
        except ValueError:
            self.assertTrue(True) # OK
        except:
            self.fail()

    def testSubVectorDiffVector(self):
        """Testa subtração de dois vetores elemento a elemento."""
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"a":3, "b":2, "c":1})

        result = v1 - v2
        self.assertDictEqual(result, {"a":-2, "b":0, "c":2})

    def testSubVectorDiffScalar(self):
        """Testa subtração de vetor a escalar."""
        v = Vector({"a":1, "b":2, "c":3})
        k = 10

        result = v - k
        self.assertDictEqual(result, {"a":-9, "b":-8, "c":-7})

    def testSubVectorInvalid(self):
        """
        Testa subtração de dois vetores incompatíveis.
        Deve resultar em erro.
        """
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"c":3, "d":2, "e":1})

        try:
            result = v1 - v2
            self.fail()
        except ValueError:
            self.assertTrue(True) # OK
        except:
            self.fail()

    def testMulVectorToVector(self):
        """Testa combinação de dois vetores."""
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"a":3, "b":2, "c":1})
        
        result = v1 * v2
        self.assertEqual(result, 10)

    def testMulVectorToScalar(self):
        """Testa multiplicação de vetor por escalar."""
        v = Vector({"a":1, "b":2, "c":3})
        k = 10
        
        result = v * k
        self.assertDictEqual(result, {"a":10, "b":20, "c":30})

    def testMulVectorInvalid(self):
        """
        Testa combinação de dois vetores com chaves totalmente diferentes.
        Deve resultar em erro.
        """
        v1 = Vector({"a":1, "b":2, "c":3})
        v2 = Vector({"c":3, "d":2, "e":1})

        try:
            result = v1 * v2
            self.fail()
        except ValueError:
            self.assertTrue(True) # OK
        except:
            self.fail()

    def testVectorPow2(self):
        """
        Testa quadrado do vetor.
        Deve se comportar do mesmo modo da multiplicação por ele mesmo.
        """
        v = Vector({"a":1, "b":2, "c":3})
        
        result = v.pow2() # 1 + 4 + 9 = 14
        self.assertEqual(result, 14)
        self.assertEqual(result, v*v)


    def testVectorUnaryNeg(self):
        """
        Testa negação do vetor.
        Deve se comportar do mesmo modo da multiplicação por -1 escalar.
        """
        v = Vector({"a":1, "b":2, "c":3})
        
        result = -v
        self.assertDictEqual(result, {"a":-1, "b":-2, "c":-3})
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
