import unittest
import math
from processors.classifiers import Bayes, EuclideanDistance, MinimumDistance, MinimumDistance2, Perceptron, PerceptronDelta
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
        classifier = EuclideanDistance()
        classifier.fit(test)

        expected_eq = "\sqrt{x^{2}_{1}+x^{2}_{2}+x^{2}_{3}-8x_{1}-10x_{2}-12x_{3}+77}"
        self.assertEqual(expected_eq, classifier.latex)

        self.assertAlmostEqual(math.sqrt(77), classifier.predict((0, 0, 0)))

        self.assertAlmostEqual(0, classifier.predict((4, 5, 6)))
    
    def testMaxDist(self):
        """
        Testa classificador máximo.
        """
        test = np.array((3, 4, 5))
        classifier = MinimumDistance()
        classifier.fit(test)

        expected_eq = "3x_{1}+4x_{2}+5x_{3}-25"
        self.assertEqual(expected_eq, classifier.latex)

        self.assertAlmostEqual(-25, classifier.predict((0, 0, 0)))

        self.assertAlmostEqual(25, classifier.predict((3, 4, 5)))
    
    def testDIJ(self):
        """
        Testa superfície de decisão.
        """
        versicolor = np.array((4.3, 1.3))
        setosa = np.array((1.5, 0.3))

        classifier = MinimumDistance2()
        classifier.fit(versicolor, setosa)

        expected_eq = "2.8x_{1}+x_{2}-8.92"
        self.assertEqual(expected_eq, classifier.latex)

        sample_setosa = (1.4, 0.2)
        self.assertTrue(classifier.predict(sample_setosa) < 0)
    
    def testPerceptron(self):
        """
        Testa geração da superfície de decisão pelo algoritmo Perceptron.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        classifier = Perceptron(1, 10_000)
        classifier.fit(cl1, cl2)

        self.assertEqual("-2x_{1}+1", classifier.latex)
        self.assertEqual(10, classifier.iterations)
        self.assertTrue(classifier.predict([0, 0]) > 0)
        self.assertTrue(classifier.predict([0, 1]) > 0)
        self.assertTrue(classifier.predict([1, 0]) < 0)
        self.assertTrue(classifier.predict([1, 1]) < 0)
    
    def testPerceptronDelta(self):
        """
        Testa geração da superfície de decisão pelo algoritmo Perceptron.
        """
        cl1 = [np.array((0, 0)), np.array((0, 1))]
        cl2 = [np.array((1, 0)), np.array((1, 1))]
        classifier = PerceptronDelta(0.5, 10_000)
        classifier.fit(cl1, cl2)

        self.assertEqual('-1.6875x_{1}+0.15625x_{2}+0.84375', classifier.latex)
        self.assertEqual(11, len(classifier.errors))
        self.assertTrue(classifier.predict([0, 0]) > 0)
        self.assertTrue(classifier.predict([0, 1]) > 0)
        self.assertTrue(classifier.predict([1, 0]) < 0)
        self.assertTrue(classifier.predict([1, 1]) < 0)
    
    def testBayes(self):
        """
        Testa bayes.
        """
        c1 = [
            (1, 0, 1),
            (1, 0, 0),
            (0, 0, 0),
            (1, 1, 0)
        ]
        
        c2 = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
            (1, 1, 1)
        ]

        result = Bayes()
        result.fit(c1, c2, 0.5, 0.5)
        self.assertEquals(result.latex, "8x_{1}-8x_{2}-8x_{3}+4")
        self.assertTrue(result.predict(c1[0]) > 0)
        self.assertTrue(result.predict(c1[1]) > 0)
        self.assertTrue(result.predict(c1[2]) > 0)
        self.assertTrue(result.predict(c1[3]) > 0)

        self.assertTrue(result.predict(c2[0]) < 0)
        self.assertTrue(result.predict(c2[1]) < 0)
        self.assertTrue(result.predict(c2[2]) < 0)
        self.assertTrue(result.predict(c2[3]) < 0)
