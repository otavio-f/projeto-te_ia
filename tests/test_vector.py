"""
Testes de unidade para módulo vetor
"""

import math
import unittest
from vector.vector import Vector


class VectorTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de vetores
    """

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
        self.assertEqual(result, [11, 12, 13])

    def testSubVectorDiffVector(self):
        """Testa subtração de dois vetores elemento a elemento."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])

        result = v1 - v2
        self.assertEqual(result, [-3, -3, -3])

    def testSubVectorDiffScalar(self):
        """Testa subtração de vetor a escalar."""
        v = Vector([1, 2, 3])
        k = 10

        result = v - k
        self.assertEqual(result, [-9, -8, -7])

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
        self.assertEqual(result, [-1, -2, -3])
        self.assertEqual(result, v * -1)


if __name__ == "__main__":
    unittest.main()
