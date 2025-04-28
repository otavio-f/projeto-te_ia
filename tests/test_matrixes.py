import unittest
from processors.evaluators import ConfusionMatrix


SAMPLE = [
    [140, 20, 0, 0],
    [10, 130, 0, 0],
    [5, 0, 150, 10],
    [15, 10, 0, 120]
]

CLASSES = ("w1", "w2", "w3", "w4")


class ConfusionMatrixTestCase(unittest.TestCase):
    """
    Classe de teste de unidade de matriz de confusão
    """

    def testPartials(self):
        "Testa as somas parciais da matriz de confusão."
        cm = ConfusionMatrix(SAMPLE, CLASSES)

        self.assertListEqual([160, 140, 165, 145], cm.M("i+").tolist())
        self.assertEquals(160, cm.M("0+"))
        self.assertEquals(140, cm.M("1+"))
        self.assertEquals(165, cm.M("2+"))
        self.assertEquals(145, cm.M("3+"))

        self.assertEquals([170, 160, 150, 130], cm.M("+i").tolist())
        self.assertEquals(170, cm.M("+0"))
        self.assertEquals(160, cm.M("+1"))
        self.assertEquals(150, cm.M("+2"))
        self.assertEquals(130, cm.M("+3"))

    def testAcerto(self):
        "Testa os acertos casuais/gerais."
        cm = ConfusionMatrix(SAMPLE, CLASSES)
        self.assertAlmostEquals(0.885, cm.Ag, delta=0.001)
        self.assertAlmostEquals(0.25, cm.Aa, delta=0.01)

    def testMetricas(self):
        "Testa as métricas de tau e kappa."
        cm = ConfusionMatrix(SAMPLE, CLASSES)

        self.assertAlmostEquals(0.846, cm.tau, delta=0.001)
        self.assertAlmostEquals(0.0002224, cm.tau_var, delta=0.0000001)
        self.assertAlmostEquals(0.846, cm.kappa, delta=0.001)
        self.assertAlmostEquals(0.014089, cm.kappa_var, delta=0.000001)
