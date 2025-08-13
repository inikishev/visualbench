from typing import Literal

import torch
from torch import nn

from ..benchmark import Benchmark, _sum_of_squares


def _human_heart_dipole_data(variant: Literal[1,2,3,4,5]):
    # values are from
    # https://github.com/jacobwilliams/MINPACK-2/blob/master/tprobs/dhhdfj.f
    if variant == 1:
        summx = 0.485
        summy = -0.0019
        suma = -0.0581
        sumb = 0.015
        sumc = 0.105
        sumd = 0.0406
        sume = 0.167
        sumf = -0.399
    elif variant == 2:
        summx = -0.69
        summy = -0.044
        suma = -1.57
        sumb = -1.31
        sumc = -2.65
        sumd = 2.0
        sume = -12.6
        sumf = 9.48
    elif variant == 3:
        summx = -0.816
        summy = -0.017
        suma = -1.826
        sumb = -0.754
        sumc = -4.839
        sumd = -3.259
        sume = -14.023
        sumf = 15.467
    elif variant == 4:
        summx = -0.809
        summy = -0.021
        suma = -2.04
        sumb = -0.614
        sumc = -6.903
        sumd = -2.934
        sume = -26.328
        sumf = 18.639
    elif variant == 5:
        summx = -0.807
        summy = -0.021
        suma = -2.379
        sumb = -0.364
        sumc = -10.541
        sumd = -1.961
        sume = -51.551
        sumf = 21.053
    else:
        raise ValueError(variant)
    return summx, summy, suma, sumb, sumc, sumd, sume, sumf


class HumanHeartDipole(Benchmark):
    """Human heart dipole objective from MINPACK2.

    The are 5 variants, of which 4th is the hardest and is selected as the default one.

    This is a least squares objective, it can be set to return a vector of residuals by calling ``benchmark.set_multiobjective()``.
    A function to combine residuals can be set via ``benchmark.set_multiobjective_func``. It is sum of squares by default.

    Doesn't support rendering.
    """
    def __init__(self, variant:Literal[1,2,3,4,5]=4):
        super().__init__(seed=0)
        self.x = nn.Parameter(torch.randn(8, generator=self.rng.torch()))
        self.data = _human_heart_dipole_data(variant)
        self.set_multiobjective_func(_sum_of_squares)

    def get_loss(self):
        x1,x2,x3,x4,x5,x6,x7,x8 = self.x.clone() # not cloning breaks hvps
        omx, omy, oA, oB, oC, oD, oE, oF = self.data

        f1 = x1 + x2 - omx
        f2 = x3 + x4 - omy
        f3 = x5*x1 + x6*x2 - x7*x3 - x8*x4 - oA
        f4 = x7*x1 + x8*x2 + x5*x3 + x6*x4 - oB

        t1 = x5**2 - x7**2
        t2 = x6**2 - x8**2
        x5x7 = x5*x7
        x6x8 = x6*x8
        f5 = x1*t1 - 2*x3*x5x7 + x2*t2 - 2*x4*x6x8 - oC
        f6 = x3*t1 + 2*x1*x5x7 + x4*t2 + 2*x2*x6x8 - oD

        t3 = x5*(x5**2 - 3*x7**2)
        t4 = x7*(x7**2 - 3*x5**2)
        t5 = x6*(x6**2 - 3*x8**2)
        t6 = x8*(x8**2 - 3*x6**2)
        f7 = x1*t3 + x3*t4 + x2*t5 + x4*t6 - oE
        f8 = x3*t3 - x1*t4 + x4*t5 - x2*t6 - oF

        return torch.stack([f1,f2,f3,f4,f5,f6,f7,f8])


# based on this https://github.com/jacobwilliams/MINPACK-2/blob/master/tprobs/dcpffj.f
def propane_combustion(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    P = 40.0
    RR = 10.0
    K5 = 1.930e-1
    K6 = 2.597e-3
    K7 = 3.448e-3
    K8 = 1.799e-5
    K9 = 2.155e-4
    K10 = 3.846e-5
    x1 = x1.abs(); x2 = x2.abs(); x3 = x3.abs(); x4 = x4.abs(); x11=x11.abs()
    pdx = P / x11
    sqpdx = torch.sqrt(pdx)

    xtau = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10

    f1 = x1 + x4 - 3.0
    f2 = 2.0*x1 + x2 + x4 + x7 + x8 + x9 + 2.0*x10 - RR
    f3 = 2.0*x2 + 2.0*x5 + x6 + x7 - 8.0
    f4 = 2.0*x3 + x9 - 4.0*RR
    f5 = K5 * x2 * x4 - x1 * x5
    f6 = K6 * torch.sqrt(x2 * x4) - torch.sqrt(x1) * x6 * sqpdx
    f7 = K7 * torch.sqrt(x1 * x2) - torch.sqrt(x4) * x7 * sqpdx
    f8 = K8 * x1 - x4 * x8 * pdx
    f9 = K9 * x1 * torch.sqrt(x3) - x4 * x9 * sqpdx
    f10 = K10 * x1**2 - x4**2 * x10 * pdx
    f11 = x11 - xtau

    return torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])

class PropaneCombustion(Benchmark):
    """Propane combustion objective from MINPACK2.

    This is a least squares objective, it can be set to return a vector of residuals by calling ``benchmark.set_multiobjective()``.
    A function to combine residuals can be set via ``benchmark.set_multiobjective_func``. It is sum of squares by default.

    Doesn't support rendering.
    """
    def __init__(self):
        super().__init__(seed=0)
        self.x = nn.Parameter(torch.randn(11, generator=self.rng.torch()))
        self.set_multiobjective_func(_sum_of_squares)

    def get_loss(self):
        fs = propane_combustion(*self.x.clone())
        return fs