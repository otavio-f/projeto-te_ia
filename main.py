"""
Reconhecimento com supervisão
"""
from vector.vector import Vector
from classifier.perceptron import Perceptron


if __name__ == "__main__":
    c1 = [Vector((0, 0)), Vector((0, 1))]
    c2 = [Vector((1, 0)), Vector((1, 1))]
    z = Perceptron("test", c1, c2)
    eq, eqrepr, iters = z.train()
    print(eq, eqrepr, iters)
    print(eq([1, 0]))