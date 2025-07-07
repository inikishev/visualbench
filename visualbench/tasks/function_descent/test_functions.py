import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch

from ...utils import totensor
from ...utils.relaxed_multikey_dict import RelaxedMultikeyDict

TEST_FUNCTIONS:"RelaxedMultikeyDict[TestFunction]" = RelaxedMultikeyDict()

def _to(self: "FunctionTransform | TestFunction", device=None, dtype=None):
    c = copy.copy(self)
    for k,v in c.__dict__.items():
        if isinstance(v, (torch.Tensor, FunctionTransform, TestFunction)):
            setattr(c, k, v.to(device=device, dtype=dtype))
    return c

class FunctionTransform(ABC):
    def transform_parameters(self, x:torch.Tensor, y:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y

    def transform_value(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def transform_point(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """where does a new point end up on the transformed function, this is inverse of transform_parameters"""
        return None

    def transform_domain(self, xmin, xmax, ymin, ymax) -> Any:
        mins = self.transform_point(xmin, ymin) # pylint:disable=assignment-from-none
        maxs = self.transform_point(xmax, ymax) # pylint:disable=assignment-from-none
        if mins is None or maxs is None: return xmin, xmax, ymin, ymax
        return [mins[0], maxs[0], mins[1], maxs[1]]

    def to(self, device=None, dtype=None):
        return _to(self, device=device, dtype=dtype)

class Shift(FunctionTransform):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def transform_parameters(self, x, y):
        return x + self.x, y + self.y

    def transform_point(self, x, y):
        return x - self.x, y - self.y

class Scale(FunctionTransform):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def transform_parameters(self, x, y):
        return x * self.x, y * self.y

    def transform_point(self, x, y):
        return x / self.x, y / self.y

    def transform_domain(self, xmin, xmax, ymin, ymax):
        min_scale = min(self.x, self.y)
        return [i/min_scale for i in (xmin, xmax, ymin, ymax)]

class Lambda(FunctionTransform):
    def __init__(
        self,
        xy: Callable[[torch.Tensor,torch.Tensor],tuple[torch.Tensor,torch.Tensor]] | None = None,
        v: Callable[[torch.Tensor],torch.Tensor] | None = None
    ):
        self.xy = xy
        self.v = v

    def transform_parameters(self, x, y):
        if self.xy is None: return x, y
        return self.xy(x, y)

    def transform_value(self, value):
        if self.v is None: return value
        return self.v(value)

    def transform_domain(self, xmin, xmax, ymin, ymax):
        return xmin, xmax, ymin, ymax

class TestFunction(ABC):

    @abstractmethod
    def objective(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        ...

    def x0(self) -> Sequence | torch.Tensor:
        ...

    @abstractmethod
    def domain(self) -> Sequence[float]:
        ...

    @abstractmethod
    def minima(self) -> Sequence[float] | torch.Tensor | None:
        ...

    def register(self, *names):
        TEST_FUNCTIONS[names] = self
        return self

    def __call__(self, x:torch.Tensor, y:torch.Tensor):
        return self.objective(x, y)

    def to(self, device=None, dtype=None) -> "TestFunction":
        return _to(self, device=device, dtype=dtype) # pyright:ignore[reportReturnType]

    def transformed(self, transforms: FunctionTransform | Sequence[FunctionTransform]):
        return TransformedFunction(self, transforms=transforms)

    def shifted(self, x, y):
        return self.transformed(Shift(x, y))

    def scaled(self, x, y):
        return self.transformed(Scale(x, y))

    def xy_tfm(self, fn: Callable[[torch.Tensor,torch.Tensor],tuple[torch.Tensor,torch.Tensor]]):
        return self.transformed(Lambda(xy=fn))

    def x_tfm(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        xy_fn = lambda x,y: (fn(x),y)
        return self.transformed(Lambda(xy=xy_fn))

    def y_tfm(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        xy_fn = lambda x,y: (x,fn(y))
        return self.transformed(Lambda(xy=xy_fn))

    def value_tfm(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        return self.transformed(Lambda(v=fn))

    def sqrt(self):
        return self.value_tfm(torch.sqrt)

    def pow(self, p):
        return self.value_tfm(lambda x: torch.pow(x, p))

    def logexp(self, u=1e-1):
        return self.value_tfm(lambda x: torch.log(u + torch.exp(x)))

    def logadd(self, u=1e-1):
        return self.value_tfm(lambda x: torch.log(x + u))

    def divadd(self, k=1.):
        return self.value_tfm(lambda x: x / (x+k))

    def muladd(self, k=-1.):
        return self.value_tfm(lambda x: x * (x+k))

class TransformedFunction(TestFunction):
    def __init__(self, function: TestFunction, transforms: FunctionTransform | Sequence[FunctionTransform]):
        self.function = function
        if isinstance(transforms, FunctionTransform): transforms = [transforms]
        self.transforms = transforms

    def objective(self, x, y):
        for tfm in self.transforms:
            x,y = tfm.transform_parameters(x, y)

        value = self.function(x, y)
        for tfm in self.transforms:
            value = tfm.transform_value(value)

        return value

    def x0(self):
        x0 = totensor(self.function.x0())
        x, y = x0
        for tfm in self.transforms:
            ret = tfm.transform_point(x, y)
            if ret is not None: x, y = ret
        return (x, y)

    def domain(self):
        domain = totensor(self.function.domain())
        xmin,xmax, ymin,ymax = domain
        for tfm in self.transforms:
            ret = tfm.transform_domain(xmin,xmax,ymin,ymax)
            if ret is not None: xmin,xmax,ymin,ymax = ret
        return (float(xmin),float(xmax), float(ymin),float(ymax))

    def minima(self):
        minima = self.function.minima()
        if minima is None: return minima

        x, y = totensor(minima)
        for tfm in self.transforms:
            ret = tfm.transform_point(x, y)
            if ret is not None: x, y = ret
        return (float(x), float(y))


class PowSum(TestFunction):
    def __init__(self, xpow, ypow, cross_add=1.0, cross_mul=0.0, abs:bool = True, post_pow = 1.0, x0=(-9,-7)):
        self.xpow, self.ypow = xpow, ypow
        self.cross_add, self.cross_mul = cross_add, cross_mul
        self.abs = abs
        self.post_pow = post_pow
        self._x0 = x0

    def objective(self, x, y):
        x = x ** self.xpow
        y = y ** self.ypow

        if self.abs:
            x = torch.abs(x)
            y = torch.abs(y)

        res = (x + y) * self.cross_add + (x * y * self.cross_mul)
        return res ** self.post_pow

    def x0(self): return self._x0
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

cross25 = PowSum(xpow=0.25, ypow=0.25).shifted(1,-2).register('cross25')
cross = PowSum(xpow=0.5, ypow=0.5).shifted(1,-2).register('cross')
cone = PowSum(xpow=1, ypow=1).shifted(1,-2).register('cone')
sphere = PowSum(xpow=2, ypow=2).shifted(1,-2).register('sphere')
convex3 = PowSum(xpow=3, ypow=3).shifted(1,-2).register('convex3')
convex4 = PowSum(xpow=4, ypow=4).shifted(1,-2).register('convex4')
convex5 = PowSum(xpow=5, ypow=5).shifted(1,-2).register('convex5')
convex32 = PowSum(xpow=3, ypow=2).shifted(1,-2).register('convex32')
convex43 = PowSum(xpow=4, ypow=3).shifted(1,-2).register('convex43')
convex405 = PowSum(xpow=4, ypow=0.5).shifted(1,-2).register('convex405')
conepow2 = PowSum(xpow=1, ypow=1, post_pow=2).shifted(1,-2).register('conepow2')
crosspow2 = PowSum(xpow=0.5, ypow=0.5, post_pow=2).shifted(1,-2).register('crosspow2')
cross25pow4 = PowSum(xpow=0.5, ypow=0.5, post_pow=4).shifted(1,-2).register('cross25pow4')
convex4pow25 = PowSum(xpow=4, ypow=4, post_pow=0.25).shifted(1,-2).register('convex4pow25')
convex96pow025 = PowSum(xpow=9, ypow=6, post_pow=0.25).shifted(1,-2).register('convex96pow025')
stretched_sphere = PowSum(2, 2, x0=(-9, -70)).scaled(1, 10).shifted(1, -2).register('stretched')


class Rosenbrock(TestFunction):
    def __init__(self, a = 1., b = 100, post_fn=torch.square):
        self.a = a
        self.b = b
        self.post_fn = post_fn

    def objective(self, x, y):
        return self.post_fn(self.a - x) + self.post_fn(self.b * (y - x**2))

    def x0(self): return (-1.1, 2.5)
    def domain(self): return (-2, 2, -1, 3)
    def minima(self): return (1, 1)

rosenbrock = Rosenbrock().register('rosen', 'rosenbrock')
rosenbrock_abs = Rosenbrock(post_fn=torch.abs).register('rosen_abs', 'rosenbrock_abs')
rosenbrock10 = Rosenbrock(b=10).register('rosen10', 'rosenbrock10')


class Rosenmax(Rosenbrock):
    def objective(self, x, y):
        return torch.maximum(self.post_fn(self.a - x), self.post_fn(self.b * (y - x**2)))

rosenmax = Rosenmax().register('rosenmax')
rosenmax_abs = Rosenmax(post_fn=torch.abs).register('rosenmaxabs', 'rosenabsmax')


class Rastrigin(TestFunction):
    def __init__(self, A=10):
        self.A = A


    def objective(self, x, y):
        return self.A * 2 + x ** 2 - self.A * torch.cos(2 * torch.pi * x) + y ** 2 - self.A * torch.cos(2 * torch.pi * y)

    def x0(self): return (-4.5, 4.3)
    def domain(self): return (-5.12, 5.12, -5.12, 5.12)
    def minima(self): return (0, 0)

rastrigin = Rastrigin().register('rastrigin')
rastrigin_shifted = Rastrigin().shifted(0.5, -1.33).register('rastrigin_shifted')

class Ackley(TestFunction):
    def __init__(self, a=20., b=0.2, c=2 * torch.pi, domain=6):
        self.a = a
        self.b = b
        self.c = c
        self.domain_ = domain


    def objective(self, x, y):
        return -self.a * torch.exp(-self.b * torch.sqrt((x ** 2 + y ** 2) / 2)) - torch.exp(
            (torch.cos(self.c * x) + torch.cos(self.c * y)) / 2) + self.a + torch.exp(torch.tensor(1, dtype=x.dtype, device=x.device))

    def x0(self): return (-self.domain_ + self.domain_ / 100, self.domain_ - self.domain_ / 95)
    def domain(self): return (-self.domain_, self.domain_, -self.domain_, self.domain_)
    def minima(self): return (0,0)

ackley = Ackley().register('ackley')
ackley_shifted = Ackley().shifted(0.5, -1.33).register('ackley_shifted')

class Beale(TestFunction):
    def __init__(self, a=1.5, b=2.25, c=2.625):
        self.a = a
        self.b = b
        self.c = c

    def objective(self, x, y):
        return (self.a - x + x * y) ** 2 + (self.b - x + x * y ** 2) ** 2 + (self.c - x + x * y ** 3) ** 2

    def x0(self): return (-4, -4)
    def minima(self): return (3, 0.5)
    def domain(self): return (-4.5, 4.5, -4.5, 4.5)

beale = Beale().register('beale')

class Booth(TestFunction):
    def objective(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def x0(self): return (0, -8)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (1, 3)

booth = Booth().register('booth')

class GoldsteinPrice(TestFunction):
    def objective(self, x,y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    def x0(self): return (-2.9, -1.9)
    def domain(self): return (-3, 3, -3, 3)
    def minima(self): return (0, -1)

golstein_price = GoldsteinPrice().register('goldstein_price')



class Norm(TestFunction):
    def __init__(self, ord:int|float=2):
        self.ord = ord

    def objective(self, x, y):
        return torch.linalg.vector_norm(torch.stack([x, y]), ord = self.ord, dim = 0) # pylint:disable=not-callable

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0,0)


l2 = Norm(2).shifted(1,-2).register('l2')
l1 = Norm(1).shifted(1,-2).register('l1')
l3 = Norm(3).shifted(1,-2).register('l3')
linf = Norm(float('inf')).shifted(1,-2).register('linf')
l0 = Norm(0).shifted(1,-2).register('l0')

class DotProduct(TestFunction):
    def __init__(self, target = (1., -2.)):
        self.target:torch.Tensor = totensor(target)

    def objective(self, x, y):
        preds = torch.stack([x, y])
        target = self.target
        while target.ndim < preds.ndim: target = target.unsqueeze(-1)
        return (preds * target.expand_as(preds)).abs().sum(0)

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return self.target

dot = DotProduct().register('dot')


class Exp(TestFunction):
    def __init__(self, base: float = torch.e): # pylint:disable=redefined-outer-name
        self.base = totensor(base)

    def objective(self, x, y):
        X = torch.stack([x, y])
        return (self.base.expand_as(X) ** X.abs()).abs().mean(0)

    def x0(self): return (-7, -9)
    def domain(self): return (-10,10,-10,10)
    def minima(self): return (0, 0)

exp = Exp().shifted(1,-2).register('exp')


class Eggholder(TestFunction):
    def __init__(self):
        super().__init__()


    def objective(self, x, y):
        return (-(y + 47) * torch.sin((y + x/2 + 47).abs().sqrt()) - x * torch.sin((x - (y + 47)).abs().sqrt())) + 959.6407
    def x0(self): return (0, 0)
    def domain(self): return (-512, 512, -512, 512)
    def minima(self): return (512, 404.2319)

eggholder = Eggholder().register('eggholder')


class PotentialWell(TestFunction):
    def objective(self, x, y):
        return (x**2 + y**2) * (1 + 0.5 * torch.sin(10 * x) * torch.sin(10 * y)) + 10 * torch.relu(x + y - 2)

    def x0(self): return (1, 3.3)
    def domain(self): return (-4, 2, -1, 5)
    def minima(self): return None
potential_well = PotentialWell().shifted(1,-2).register('potential_well')


class DoubleWell(TestFunction):
    def __init__(self,a=1.5, b=0.5):
        super().__init__()
        self.a = a; self.b = b

    def objective(self, x, y):
        return (x**2 - self.a**2)**2 + 0.5*x*y + (y**2 - self.b**2)**2

    def x0(self): return (-0.01, 0.01)
    def domain(self): return (-2.5, 2.5, -2.5, 2.5)
    def minima(self): return None
double_well = DoubleWell().register('double_well')


class DipoleField(TestFunction):
    """Magnetic Dipole Interaction Field"""
    def objective(self, x, y):
        eps = 1e-3
        term1 = -(x-1)/(((x-1)**2 + y**2 + eps)**1.5)
        term2 = -(x+1)/(((x+1)**2 + y**2 + eps)**1.5)
        return term1 + term2 + 0.1*(x**2 + y**2)


    def x0(self): return (0., 1.)
    def domain(self): return (-2, 2, -2, 2)
    def minima(self): return None
dipole_field = DipoleField().register('dipole_field', 'dipole')


class ChaoticPotential(TestFunction):
    def objective(self, x, y):
        term1 = torch.sin(3*x) * torch.cos(4*y) * torch.exp(-0.1*(x**2 + y**2))
        term2 = 2*torch.abs(torch.sin(2*x) + torch.cos(3*y))
        term3 = 0.5*torch.relu(x**2 - y**2 - 1)
        return term1 + term2 + term3


    def x0(self): return (-3.3, 3.)
    def domain(self): return (-4, 4, -4, 4)
    def minima(self): return None
chaotic_potential = ChaoticPotential().register('chaotic_potential')


class PlasmaSurface(TestFunction):
    def objective(self, x,y):
        return (x**4 + y**4) - 3*(x**2 + y**2) + 2*torch.abs(x*y) + 0.5*torch.sin(8*x)*torch.sin(8*y)

    def x0(self): return (-0.1, 0.1)
    def domain(self): return (-2, 2, -2, 2)
    def minima(self): return None
plasma_surface = PlasmaSurface().register('plasma_surface')


class QuantumTunneling(TestFunction):
    def objective(self, x, y):
        base = (x**2 - 1)**2 + (y**2 - 1)**2
        noise = 0.2*torch.sin(15*x)*torch.sin(15*y)
        barrier = 2*torch.relu(x**2 + y**2 - 0.5)
        return base + noise + barrier

    def x0(self): return (-1.2, 2.)
    def domain(self): return (-2, 2, -2, 2)
    def minima(self): return None
quantum_tunneling = QuantumTunneling().register('quantum_tunneling')


class ChemicalPotential(TestFunction):
    def objective(self, x, y):
        terms = [
            (-200, -1, 0, -10, 1, 0),
            (-100, -1, 0, -10, 0, 0.5),
            (-170, -6.5, 11, -6.5, -0.5, 1.5),
            (15, 0.7, 0.6, 0.7, -1, 1)
        ]
        V = torch.zeros_like(x)
        for A, a, b, c, x0, y0 in terms:
            V += A * torch.exp(a*(x-x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0)**2)
        return V

    def x0(self): return (-1.25, -0.25)
    def domain(self): return (-1.5, 1.0, -0.5, 2.0)
    def minima(self): return None
chemical_potential = ChemicalPotential().register('chemical_potential')



class FitnessLandscape(TestFunction):
    def objective(self, x, y):
        return (x**2 - 4*x + y**2 - 6*y +
                torch.cos(3*x) + torch.sin(2*y) +
                0.5*x*y + 13)

    def x0(self): return (4, 0.)
    def domain(self): return (-3, 6, -2, 7)
    def minima(self): return None
fitness_landscape = FitnessLandscape().register('fitness_landscape', 'fitness')


class ArmError(TestFunction):
    """Minimize positioning error for a 2-link robotic arm reaching a target."""
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = 2.0, 1.0  # Link lengths
        self.target = torch.tensor([2.0, 1.0])  # Target position

    def objective(self, x, y):
        x = self.l1 * torch.cos(x) + self.l2 * torch.cos(x + y)
        y = self.l1 * torch.sin(x) + self.l2 * torch.sin(x + y)
        return (x - self.target[0])**2 + (y - self.target[1])**2

    def x0(self): return (-2, 0.)
    def domain(self): return (-torch.pi, torch.pi, -torch.pi, torch.pi)
    def minima(self): return None
arm_error = ArmError().register('arm_error')


class Spiral(TestFunction):
    """outward spiral"""
    def __init__(self, length=17.0, center_intensity=1.0, max_spiral_intensity=0.9, r_stop=0.9, blend_start_ratio=0.9):
        super().__init__()
        self.length = length
        self.center_intensity = center_intensity
        self.max_spiral_intensity = max_spiral_intensity
        self.r_stop = r_stop
        self.blend_start_ratio = blend_start_ratio

    def objective(self, x, y):
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        spiral_angle = theta - r * self.length

        blend_start_radius = self.r_stop * self.blend_start_ratio

        condition_increase = r <= blend_start_radius
        condition_blend = (r > blend_start_radius) & (r <= self.r_stop)
        condition_stop = r > self.r_stop

        radial_intensity_increase = self.max_spiral_intensity * (r / (blend_start_radius))
        radial_intensity_blend = self.max_spiral_intensity * (1 - (r - blend_start_radius) / (self.r_stop - blend_start_radius))
        radial_intensity_stop = torch.zeros_like(r)

        spiral_radial_intensity = torch.where(condition_stop, radial_intensity_stop,
                                        torch.where(condition_blend, radial_intensity_blend,
                                                    torch.where(condition_increase, radial_intensity_increase, torch.zeros_like(r))))

        spiral_intensity = spiral_radial_intensity * (0.5 * torch.cos(spiral_angle) + 0.5)
        intensity = self.center_intensity - spiral_intensity

        return intensity

    def x0(self): return (0.09, 0.05)
    def domain(self): return (-1, 1, -1, 1)
    def minima(self): return None

spiral = Spiral().register('spiral')


class LogSumExp(TestFunction):
    def __init__(self, A=1.0, B=1.0, k=1.0):
        super().__init__()
        self.A = A
        self.B = B
        self.k = k

    def objective(self, x, y):

        term1 = self.k * self.A * x**2
        term2 = self.k * self.B * y**2

        terms = torch.stack([term1, term2])
        value = (1.0 / self.k) * torch.logsumexp(terms, dim=0)

        return value.view_as(x)

    def x0(self): return (5, -6)
    def domain(self): return (-8,8, -8, 8)
    def minima(self): return (0, 0)

logsumexp = LogSumExp().shifted(1, -2).register('logsumexp')

class Stadium(TestFunction):
    def __init__(self, half_width: float = 10, alpha: float = 0.5):
        super().__init__()
        self.half_width = half_width
        self.alpha = alpha

    def objective(self, x, y):
        x_clamped = torch.clamp(x, min=-self.half_width, max=self.half_width)
        dist_sq = torch.pow(x - x_clamped, 2) + torch.pow((y+1.5), 2)
        gradient_term = -self.alpha * x
        value = dist_sq + gradient_term
        return value

    def x0(self): return (-4, 2.5)
    def domain(self): return (-5,15, -10,10)
    def minima(self): return None

stadium = Stadium().register('stadium')

class Around(TestFunction):
    def objective(self,x,y):
        return torch.atan2(x,abs(y)) + (0.02*x)**2
    def x0(self): return (8, 0.1)
    def domain(self): return (-20, 10, -15, 15)
    def minima(self): return None

around = Around().register('around')


class Tanh(TestFunction):
    def __init__(self, xpow=2, ypow=2):
        self.xpow = xpow
        self.ypow = ypow
    def objective(self,x,y):
        return (x.tanh()**self.xpow) + (y.tanh()**self.ypow)

    def x0(self): return (4, -2.5)
    def domain(self): return (-5, 5, -5, 5)
    def minima(self): return None

tanh = Tanh().shifted(-1, 2).register('tanh')

class IllConditioned(TestFunction):
    def __init__(self, b = 1e-4):
        self.b = b

    def objective(self,x,y):
        return x**2 + y**2 + (2-self.b) * x * y

    def x0(self): return (-9, 2.5)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

ill_conditioned = IllConditioned().shifted(-1, 2).register('ill_conditioned', 'ill')
ill_pseudoconvex = IllConditioned().divadd(0.1).shifted(-1, 2).register('ill_pseudoconvex', 'illpc')
very_ill_conditioned = IllConditioned(1e-6).shifted(-1, 2).register('very_ill_conditioned', 'very_ill')



class IllPiecewise(TestFunction):
    def __init__(self, b = 1e-4):
        self.b = b

    def objective(self, x, y):
        return x.abs().maximum(y.abs()) + (1/self.b)*(x + y).abs()

    def x0(self): return (-9, 2.5)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

ill_piecewise = IllPiecewise().shifted(-1, 2).register('ill_piecewise', 'piecewise', 'illp')
ill_piecewise_pseudoconvex = IllPiecewise().shifted(-1, 2).register('illppc')

class IllSqrt(TestFunction):
    def __init__(self, b = 1e-4):
        self.b = b

    def objective(self, x, y):
        z = (2-self.b)*(x*y)
        return (x**2+y**2)**0.5 + z.abs().sqrt().copysign(z)

    def x0(self): return (-15,-5)
    def domain(self): return (-20, 20, -20, 20)
    def minima(self): return (0, 0)

ill_sqrt = IllSqrt().shifted(-1, 2).register('ill_sqrt', 'ills')


class Dice(TestFunction):
    def __init__(self, eps = 1e-8):
        self.eps = eps

    def objective(self, x, y):
        _dice = lambda x, y: 1 - (2 * (x.sigmoid() * y.sigmoid()) + self.eps) / (x.sigmoid() + y.sigmoid() + self.eps)
        return _dice(x,y) + _dice(-x,-y) + _dice(x, -y) + _dice(-x, y)

    def x0(self): return (-9, 8.5)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

dice = Dice().shifted(-1,2).register('dice')

class IOU(TestFunction):
    def __init__(self, eps = 1e-8):
        self.eps = eps

    def objective(self, x, y):
        _iou = lambda x, y: 1 - ((x.sigmoid() * y.sigmoid()) + self.eps) / (((x.sigmoid() + y.sigmoid()) -( x.sigmoid() * y.sigmoid())) + self.eps)
        return _iou(x,y) + _iou(-x,-y) + _iou(x, -y) + _iou(-x, y)

    def x0(self): return (-9, 5)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

iou = IOU().shifted(-1,2).register('iou')

class LeastSquares(TestFunction):
    def objective(self, x, y):
        return (2*x + 3*y - 5)**2 + (5*x - 2*y - 3)**2

    def x0(self): return (-0.9, 0)
    def domain(self): return (-1,3,-1,3)
    def minima(self): return (1, 1)
least_squares = LeastSquares().register('least_squares', 'lstsq')



class Star(TestFunction):
    def __init__(self, post_fn = torch.square, max: bool = False):
        super().__init__()
        self.post_fn = post_fn
        self.max = max

    def objective(self, x, y):
        f1 = self.post_fn(x - 6)
        f2 = self.post_fn(y - 2e-1)
        f3 = self.post_fn(x*y - 2)
        if self.max: return f1.maximum(f2).maximum(f3)
        return f1 + f2 + f3

    def x0(self): return (-7, -8)
    def domain(self): return (-10,10,-10,10)
    def minima(self): return None

star = Star().register('star')
star_abs = Star(torch.abs).register('star_abs')
star_max = Star(torch.abs, max=True).register('star_max')

class CBarriers(TestFunction):
    def __init__(
        self,
        num_barriers: int = 4,
        outer_radius: float = 4.0,
        inner_radius: float = 1.0,
        spacing: Literal['linear', 'geometric', 'reciprocal'] = 'linear',
        barrier_height: float = 50.0,
        radial_width: float = 0.2,
        gap_sharpness: float = 0.6,
        p = 2,
    ):
        """
        Args:
            num_barriers (int): The number of nested C-barriers.
            outer_radius (float): The radius of the outermost barrier.
            inner_radius (float): The target radius of the innermost barrier.
            spacing (str): Method for spacing barriers. Can be 'linear', 'geometric',
                           or 'reciprocal'.
            barrier_height (float): The height of the potential barriers.
            radial_width (float): The width (thickness) of the C-rings.
            gap_sharpness (int): Controls the size of the entrance.
        """
        super().__init__()

        self.num_barriers = num_barriers
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.spacing = spacing
        self.barrier_height = barrier_height
        self.radial_width = radial_width
        self.gap_sharpness = gap_sharpness
        self.eps = 1e-8 # for numerical stability
        self.p = p

    def _get_radii(self) -> torch.Tensor:
        """Calculates the list of radii based on the chosen spacing method."""
        if self.num_barriers == 1:
            return torch.tensor([self.outer_radius])

        if self.spacing == 'linear':
            # Evenly spaced radii from outer to inner
            return torch.linspace(self.outer_radius, self.inner_radius, self.num_barriers)
        if self.spacing == 'geometric':
            # Radii spaced by a constant ratio
            return torch.from_numpy(np.geomspace(self.outer_radius, self.inner_radius, self.num_barriers)).float()
        if self.spacing == 'reciprocal':
            # Original method, scaled to fit the outer radius
            indices = torch.arange(self.num_barriers)
            radii = self.outer_radius / (indices + 1)
            # This method ignores inner_radius, as it's defined by the reciprocal rule
            return radii

        raise ValueError("spacing must be 'linear', 'geometric', or 'reciprocal'")

    def objective(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.p % 2 != 0: x,y = x.abs(), y.abs()
        base_potential = x**self.p + y**self.p

        r = torch.sqrt(x**2 + y**2 + self.eps)
        theta = torch.atan2(y, x)

        total_barriers = torch.zeros_like(x)
        radii = self._get_radii()

        for i, radius in enumerate(radii):
            gap_angle = i * torch.pi

            radial_term = torch.exp(-(r - radius)**2 / (2 * self.radial_width**2))
            angular_term = (torch.sin((theta - gap_angle) / 2)**2)**self.gap_sharpness

            c_barrier = self.barrier_height * radial_term * angular_term
            total_barriers += c_barrier

        return base_potential + total_barriers



    def x0(self): return (-4.5, 2)
    def domain(self): return (-5,5,-5,5)
    def minima(self): return (0, 0)

cbarriers = CBarriers().shifted(1, -2).register('cbarrier')


class Switchback(TestFunction):
    def __init__(self, narrowness=50, turn_coord=10, turn_sharpness=5):
        super().__init__()
        self.narrowness = narrowness
        self.turn_coord = turn_coord
        self.turn_sharpness = turn_sharpness

    def objective(self, x, y):
        self.narrowness = 50.0
        self.turn_coord = 10.0
        self.turn_sharpness = 5.0

        f1 = -(x + y) + self.narrowness * (x - y)**2
        f2 = -(x - y) + self.narrowness * (x + y - self.turn_coord)**2
        transition_coord = x + y - self.turn_coord
        sigmoid = 1.0 / (1.0 + torch.exp(-self.turn_sharpness * transition_coord))
        return (1.0 - sigmoid) * f1 + sigmoid * f2

    def x0(self): return (0, 17)
    def domain(self): return (-7,13,0,20)
    def minima(self): return None

switchback = Switchback().register('switchback', 'switch')
