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

    def transform_value(self, value: torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
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
        v: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor] | None = None
    ):
        self.xy = xy
        self.v = v

    def transform_parameters(self, x, y):
        if self.xy is None: return x, y
        return self.xy(x, y)

    def transform_value(self, value, x, y):
        if self.v is None: return value
        return self.v(value, x, y)

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
        return self.transformed(Lambda(v=lambda l, x, y: fn(l)))

    def xy_value_tfm(self, fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor], torch.Tensor]):
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

    def mo_func(self) -> Callable | None:
        return None

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
            value = tfm.transform_value(value, x, y)

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

    def mo_func(self):
        return self.function.mo_func()


class PowSum(TestFunction):
    def __init__(self, xpow, ypow, cross_add=1.0, cross_mul=0.0, abs:bool = True, post_pow = 1.0, x0=(-9,-7)):
        self.xpow, self.ypow = xpow, ypow
        self.cross_add, self.cross_mul = cross_add, cross_mul
        self.abs = abs
        self.post_pow = post_pow
        self._x0 = x0

    def objective(self, x, y):
        if self.abs:
            x = torch.abs(x)
            y = torch.abs(y)

        x = x ** self.xpow
        y = y ** self.ypow

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

convex32_stretched = PowSum(xpow=3, ypow=2, x0=(-9, -70)).scaled(1, 10).shifted(1,-2).register('convex32s')
stretched_sphere = PowSum(2, 2, x0=(-9, -70)).scaled(1, 10).shifted(1, -2).register('stretched')


class Rosenbrock(TestFunction):
    def __init__(self, a = 1., b = 100, pd_fn=torch.square, pd_fn2 = None, mo:bool=False):
        self.a = a
        self.b = b
        self.pd_fn = pd_fn
        if pd_fn2 is None: pd_fn2 = pd_fn
        self.pd_fn2 = pd_fn2
        self.mo = mo

    def _get_terms(self, x, y):
        term1 = self.pd_fn(self.a - x)
        term2 = self.b * self.pd_fn(y - self.pd_fn2(x))
        return term1, term2

    def objective(self, x, y):
        term1, term2 = self._get_terms(x, y)
        if self.mo: return torch.stack([term1, term2])
        return term1 + term2

    def mo_func(self):
        return lambda x: x.pow(2).sum(0)

    def x0(self): return (-1.1, 2.5)
    def domain(self): return (-2, 2, -1, 3)
    def minima(self): return (1, 1)

rosenbrock = Rosenbrock().register('rosen', 'rosenbrock')
rosenbrock_abs = Rosenbrock(pd_fn = torch.abs).register('rosen_abs',)
rosenbrock_abs2 = Rosenbrock(pd_fn = torch.abs, pd_fn2=torch.square).register('rosen_abs2')
rosenbrock_abs3 = Rosenbrock(pd_fn = torch.square, pd_fn2=torch.abs).register('rosen_abs3',)
rosenbrock10 = Rosenbrock(b=10).register('rosen10', 'rosenbrock10')

rosenbrock_mo = Rosenbrock(pd_fn=torch.abs, pd_fn2=torch.square, mo=True).register('rosen_mo', 'mo_rosen')

class ChebushevRosenbrock(TestFunction):
    def __init__(self, p=1., a = 1/4, pd_fn=torch.square, pd_fn2 = None):
        self.a = a
        self.p = p
        self.pd_fn = pd_fn
        if pd_fn2 is None: pd_fn2 = pd_fn
        self.pd_fn2 = pd_fn2

    def _get_terms(self, x, y):
        term1 = 1/4 * self.pd_fn(x - 1)
        term2 = self.p * self.pd_fn(y - 2*self.pd_fn2(x) + 1)
        return term1, term2

    def objective(self, x, y):
        term1, term2 = self._get_terms(x, y)
        return term1 + term2

    def x0(self): return (-1.2, 1.6)
    def domain(self): return (-2, 2, -2, 2)
    def minima(self): return (1, 1)

crosen = ChebushevRosenbrock().register('crosen')
crosenabs = ChebushevRosenbrock(pd_fn=torch.abs).register('crosenabs')
crosenabs2 = ChebushevRosenbrock(pd_fn=torch.abs, pd_fn2=torch.square).register('crosenabs2')
crosenabs3 = ChebushevRosenbrock(pd_fn=torch.square, pd_fn2=torch.abs).register('crosenabs3')



class Rastrigin(TestFunction):
    def __init__(self, A=10):
        self.A = A


    def objective(self, x, y):
        return self.A * 2 + x ** 2 - self.A * torch.cos(2 * torch.pi * x) + y ** 2 - self.A * torch.cos(2 * torch.pi * y)

    def x0(self): return (-4.5, 4.3)
    def domain(self): return (-5.12, 5.12, -5.12, 5.12)
    def minima(self): return (0, 0)

rastrigin = Rastrigin().shifted(0.5, -1.33).register('rastrigin')

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

ackley = Ackley().shifted(0.5, -1.33).register('ackley')

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



class DipoleField(TestFunction):
    """Magnetic Dipole Interaction Field"""
    def objective(self, x, y):
        eps = 1e-3
        term1 = -(x-1)/(((x-1)**2 + y**2 + eps)**1.5)
        term2 = -(x+1)/(((x+1)**2 + y**2 + eps)**1.5)
        return term1 + term2 + 0.1*(x**2 + y**2)


    def x0(self): return (0.3, 1.8)
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

class Around(TestFunction):
    def objective(self,x,y):
        return torch.atan2(x,abs(y)) + (0.02*x)**2
    def x0(self): return (8, 0.1)
    def domain(self): return (-20, 10, -15, 15)
    def minima(self): return None

around = Around().value_tfm(lambda x: x+1.6).register('around')


class IllConditioned(TestFunction):
    def __init__(self, b = 1e-4):
        self.b = b

    def objective(self,x,y):
        return x**2 + y**2 + (2-self.b) * x * y

    def x0(self): return (-9, 2.5)
    def domain(self): return (-10, 10, -10, 10)
    def minima(self): return (0, 0)

ill1 = IllConditioned(1e-1).shifted(-1, 2).register('ill1')
ill2 = IllConditioned(1e-2).shifted(-1, 2).register('ill2')
ill3 = IllConditioned(1e-3).shifted(-1, 2).register('ill3')
ill4 = IllConditioned(1e-4).shifted(-1, 2).register('ill', 'ill4')
ill6 = IllConditioned(1e-6).shifted(-1, 2).register('ill6')
ill_pseudoconvex = IllConditioned().divadd(0.1).shifted(-1, 2).register('ill_pseudoconvex', 'illpc')


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


class Oscillating(TestFunction):
    def __init__(self, a=1, b=8*torch.pi, c=1e-2, d=100.0):
        """
        Args:
            a (float): The constant amplitude of the oscillation.
            b (float): Controls the frequency of the oscillation.
            c (float): Controls the steepness of the valley floor, guiding towards x=0.
            d (float): Controls the height of the barrier for x <= 0.
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = 1e-9

    def x0(self): return (-2.6, -0.1)
    def domain(self): return (-3,0.025,-1.5,1.5)
    def minima(self): return (0,0)

    def objective(self, x, y):
        x = -x # left to right
        mask = x > 0
        angle = self.b / (x + self.epsilon)
        path_y = self.a * torch.sin(angle)
        term1_pos = (y - path_y)**2
        term2_pos = self.c * x**2
        val_pos = term1_pos + term2_pos
        val_neg = self.d * x**2 + y**2
        return torch.where(mask, val_pos, val_neg)

oscillating = Oscillating().shifted(1, -2).register('oscillating', 'osc')


class Mycs1(TestFunction):
    """strong curvature"""
    def objective(self, x, y):
        term1 = torch.exp(0.1 * x**2 + 0.5 * y**2)
        term2 = (x - 2*y)**2
        return term1 + term2

    def x0(self): return (3,-3.5)
    def domain(self): return (-4, 4, -4, 4)
    def minima(self): return (0, 0)

mycs1 = Mycs1().shifted(1, -2).register('mycs1')


class Mycs2(TestFunction):
    """newton oscillates"""
    def objective(self, x, y):
        term1 = torch.log(1 + torch.exp(5*x + 3*y - 1)) # Convex
        term2 = torch.log(1 + torch.exp(-x - 2*y + 1)) # Convex
        term3 = 2*x**2 + y**2 # Convex
        return term1 + term2 + term3

    def x0(self): return (0.8,0.6)
    def domain(self): return (-1, 1, -1, 1)
    def minima(self): return None

mycs2 = Mycs2().shifted(1, -2).register('mycs2')


