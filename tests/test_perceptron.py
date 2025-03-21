"""
Testes de unidade para módulo vetor
"""
import unittest
from vector.vector import Vector
from classifier.perceptron import Perceptron


class PerceptronTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de Perceptron
    """
    def testConverge(self):
        """Testa geração da superfície de decisão."""
        cl1 = [Vector((0, 0)), Vector((0, 1))]
        cl2 = [Vector((1, 0)), Vector((1, 1))]
        perceptron = Perceptron("teste", cl1, cl2)

        plane, eq, iters = perceptron.perceptron()
        self.assertEqual(eq, "-2x1 + 0x2 + 1")
        self.assertEqual(iters, 16)
        self.assertEqual(1, plane([0, 0]))