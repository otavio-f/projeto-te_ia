"""
Classe classificadora Perceptron
"""

from dataclasses import dataclass
from dataclasses import field
from vector.vector import Vector, Equation
import random

@dataclass
class Perceptron:
    name: str
    c1v: [Vector, ...] # vetores da classe 1
    c2v: [Vector, ...] # vetores da classe 2

    def perceptron(self, max_iters=1000) -> 'Callable[[float, ...], float], str, int':
        w = Vector([0 for _ in self.c1v[0]]).augment(0) # vetor aumentado [..., 0]
        iters = 0
        err = True
        while iters < max_iters and err:
            err = False
            for v in self.c1v:
                iters += 1
                yk = v.augment(1) # Vetor aumentado [..., 1]
                test = w * yk.T
                if test <= 0: # deveria ser >0
                    c = 1 # random.random() # de 0 a 1
                    w = w + (c * yk)
                    err = True
            for v in self.c2v:
                iters += 1
                yk = v.augment(1) # Vetor aumentado [..., 1]
                test = w * yk.T
                if test >= 0: # deveria ser <0
                    c = 1 # random.random() # de 0 a 1
                    w = w - (c * yk)
                    err = True
        function = lambda coords: w * Vector(coords).augment(1)
        # function = lambda coords: sum(k*x for k,x in zip(w, coords)) #5x+4y+z = w=(5,4,3), x=(1,2,9)
        wl = list(w)
        expression = f"{wl[0]}x1"
        for k in range(1,len(wl)-1):
            expression += " - " if wl[k]<0 else " + "
            expression += f"{wl[k]}x{k+1}"
        expression += " - " if wl[-1] < 0 else " + " + f"{wl[-1]}"
        return function, expression, iters

    def delta_perceptron(self) -> [Equation, int]:
        pass
