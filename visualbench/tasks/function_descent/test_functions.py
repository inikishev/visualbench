import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence


import torch
from myai.python_tools import RelaxedMultikeyDict
from myai.transforms import totensor

TEST_FUNCTIONS:"RelaxedMultikeyDict[TestFunction]" = RelaxedMultikeyDict()

class TestFunction(ABC):
    @abstractmethod
    def objective(self, __X:torch.Tensor) -> torch.Tensor:
        ...
    def x0(self) -> Sequence | torch.Tensor:
        ...
    @abstractmethod
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]] | tuple[float,float,float,float] | Sequence[Sequence[float]] | Sequence[float] | None:
        ...
    @abstractmethod
    def minima(self) -> Sequence | torch.Tensor:
        ...
    def register(self, *names):
        TEST_FUNCTIONS[names] = self
        return self

    def __call__(self, x:torch.Tensor):
        return self.objective(x)

    def to(self,device,dtype):
        c = copy.copy(self)
        for k,v in c.__dict__.items():
            if isinstance(v, torch.Tensor): setattr(c, k, v.to(device = device, dtype = dtype))
        return c


class PowSum(TestFunction):
    def __init__(self, pow, mul, add, abs=True, post_pow:float = 1):
        self.pow = totensor(pow)
        self.mul = totensor(mul)
        self.add = totensor(add)
        self.abs = abs
        self.post_pow = totensor(post_pow)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting
        x = x * self.mul + self.add
        if (self.pow < 1).any(): x = x.abs()
        x = x ** self.pow
        if self.abs: x = x.abs()
        return x.mean(-1) ** self.post_pow

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return self.add

cross = PowSum(0.5, 1, [1, -2]).register('cross')
cone = PowSum(1, 1, [1, -2]).register('cone')
sphere = PowSum(2, 1, [1, -2]).register('sphere')
convex3 = PowSum(3, 1, [1, -2]).register('convex3')
convex32 = PowSum((3, 2), 1, [1, -2]).register('convex32')
convex43 = PowSum((4, 3), 1, [1, -2]).register('convex43')
conepow2 = PowSum(1, 1, [1, -2], post_pow=2).register('conepow2', 'cone2')
crosspow2 = PowSum(0.5, 1, [1, -2], post_pow=2).register('crosspow2', 'cross2')
spherepow2 = PowSum(1, 1, [1, -2], post_pow=2).register('spherepow2', 'sphere2')
conepow05 = PowSum(1, 1, [1, -2], post_pow=0.5).register('conepow05', 'cone05')
crosspow05 = PowSum(0.5, 1, [1, -2], post_pow=0.5).register('crosspow05', 'cross05')

class Rosenbrock(TestFunction):
    def __init__(self, a = 1., b = 100.):
        self.a = a
        self.b = b

    def objective(self, X):
        x,y = X
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def x0(self): return (-1.1, 2.5)
    def domain(self): return (-2, 2), (-1, 3)
    def minima(self): return (1, 1)

rosenbrock = Rosenbrock().register('rosen', 'rosenbrock')

class Rastrigin(TestFunction):
    def __init__(self, A=10., shifts = (0,0)):
        self.A = A
        self.shifts = shifts

    def objective(self, X):
        x,y = X
        x = x + self.shifts[0]
        y = y + self.shifts[1]
        return self.A * 2 + x ** 2 - self.A * torch.cos(2 * torch.pi * x) + y ** 2 - self.A * torch.cos(2 * torch.pi * y)

    def x0(self): return (-4.5, 4.3)
    def domain(self): return (-5.12, 5.12), (-5.12, 5.12)
    def minima(self): return (0, 0)

rastrigin = Rastrigin().register('rastrigin')
rastrigin_shifted = Rastrigin(shifts = (0.5, -1.33)).register('rastrigin_shifted')

class Ackley(TestFunction):
    def __init__(self, a=20., b=0.2, c=2 * torch.pi, domain=16, shifts = (0,0)):
        self.a = a
        self.b = b
        self.c = c
        self.domain_ = domain
        self.shifts = shifts


    def objective(self, X:torch.Tensor):
        x,y = X
        x = x + self.shifts[0]
        y = y + self.shifts[1]
        return -self.a * torch.exp(-self.b * torch.sqrt((x ** 2 + y ** 2) / 2)) - torch.exp(
            (torch.cos(self.c * x) + torch.cos(self.c * y)) / 2) + self.a + torch.exp(torch.tensor(1, dtype=X.dtype, device=X.device))

    def x0(self): return (-self.domain_ + self.domain_ / 100, self.domain_ - self.domain_ / 95)
    def domain(self): return (-self.domain_, self.domain_), (-self.domain_, self.domain_)
    def minima(self): return tuple(-i for i in self.shifts)

ackley = Ackley().register('ackley')
ackley_shifted = Ackley(shifts = (0.5, -1.33)).register('ackley_shifted')

class Beale(TestFunction):
    def __init__(self, a=1.5, b=2.25, c=2.625):
        self.a = a
        self.b = b
        self.c = c

    def objective(self, X:torch.Tensor):
        x,y = X
        return (self.a - x + x * y) ** 2 + (self.b - x + x * y ** 2) ** 2 + (self.c - x + x * y ** 3) ** 2

    def x0(self): return (-4, -4)
    def minima(self): return (3, 0.5)
    def domain(self): return (-4.5, 4.5), (-4.5, 4.5)

beale = Beale().register('beale')

class Booth(TestFunction):
    def objective(self, X:torch.Tensor):
        x,y = X
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def x0(self): return (0, -8)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return (1, 3)

booth = Booth().register('booth')

class IllConditioned(TestFunction):
    def objective(self, X:torch.Tensor):
        x,y = X
        return (x + 1000 * y - 5) ** 2 + (2 * x + y - 2) ** 2

    def x0(self): return (750, -8)
    def domain(self): return ((-1000, 1000), (-10, 10))
    def minima(self): return None

ill_conditioned = IllConditioned().register('ill_conditioned', 'ill')

class GoldsteinPrice(TestFunction):
    def objective(self, X):
        x, y = X
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    def x0(self): return (-2.9, -1.9)
    def domain(self): return (-3, 3), (-3, 3)
    def minima(self): return (0, -1)

golstein_price = GoldsteinPrice().register('goldstein_price')


class NonConvex(TestFunction):
    def objective(self, X):
        x, y = X
        return (x**4 - 2*y**3 + y**4 - 2*x**2 + (torch.sin(x-y)*3)**3) + 28.708

    def x0(self): return (2.5, -3)
    def domain(self): return (-4, 4), (-4, 4)
    def minima(self): return None

non_convex = NonConvex().register('nonconvex')



class Norm(TestFunction):
    def __init__(self, ord:int|float=2, add = (1, -2)):
        self.ord = ord
        self.add = totensor(add)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        return torch.linalg.vector_norm(x + self.add, ord = self.ord, dim = -1) # pylint:disable=not-callable

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return self.add


l2 = Norm(2).register('l2')
l1 = Norm(1).register('l1')
l3 = Norm(3).register('l3')
linf = Norm(float('inf')).register('linf')
l0 = Norm(0).register('l0')

class CrossEntropy(TestFunction):
    def __init__(self, target:torch.Tensor = torch.tensor([1., 0]), logits=True,):
        self.target:torch.Tensor = target
        self.logits = logits

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        fn = torch.nn.functional.binary_cross_entropy_with_logits if self.logits else torch.nn.functional.binary_cross_entropy
        return fn(x, self.target.expand_as(x), reduction='none').mean(-1)

    def x0(self): return (-5, 9)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return self.target

bce = CrossEntropy().register('ce', 'bce')


class DotProduct(TestFunction):
    def __init__(self, target = (1., -2.)):
        self.target:torch.Tensor = totensor(target)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        return (x * self.target).abs().sum(-1)

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return self.target

dot = DotProduct().register('dot')



class CosineSimilarity(TestFunction):
    def __init__(self, target = (1., -2.), eps=1e-7):
        self.target:torch.Tensor = totensor(target)
        self.eps = eps
    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        return (x * self.target).sum(-1) / ((torch.linalg.vector_norm(x, dim=-1) * torch.linalg.vector_norm(self.target)) + self.eps)# pylint:disable=not-callable

    def x0(self): return (-9, 7)
    def domain(self): return (-10, 10), (-10, 10)
    def minima(self): return self.target

cossim = CosineSimilarity().register('cossim')


class Exp(TestFunction):
    def __init__(self, add = (1, -2), exp: float = torch.e): # pylint:disable=redefined-outer-name
        self.add = totensor(add)
        self.exp = totensor(exp)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        return (self.exp.expand_as(x) ** x.abs()).abs().mean(-1)

    def x0(self): return (2.5, -5)
    def domain(self): return (-3, 5), (-6, 4)
    def minima(self): return self.add
rexp = Exp().register('exp')



class PowMax(TestFunction):
    def __init__(self, powers=(0.5, 1, 2), add = (1/3, -2/3)):
        self.powers = totensor(powers)
        self.add = totensor(add)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        x = (x + self.add).abs()
        xs = [(x**p).abs() for p in self.powers]
        return torch.stack(xs).amax(0).mean(-1)

    def x0(self): return (-9/3, 7/3)
    def domain(self): return (-10/3, 10/3), (-10/3, 10/3)
    def minima(self): return self.add
powmax = PowMax().register('powmax')

class PowMin(TestFunction):
    def __init__(self, powers=(0.5, 1, 2), add = (1/3, -2/3)):
        self.powers = totensor(powers)
        self.add = totensor(add)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        x = (x + self.add).abs()
        xs = [(x**p).abs() for p in self.powers]
        return torch.stack(xs).amin(0).mean(-1)

    def x0(self): return (-9/3, 7/3)
    def domain(self): return (-10/3, 10/3), (-10/3, 10/3)
    def minima(self): return self.add
powmin = PowMax().register('powmin')


class PowDiff(TestFunction):
    def __init__(self, powers=(0.5, 1, 2), add = (1/3, -2/3)):
        self.powers = totensor(powers)
        self.add = totensor(add)

    def objective(self, x):
        if x.ndim > 1:
            x = x.movedim(0, -1) # needed for broadcasting

        x = (x + self.add).abs()
        xs = [(x**p).abs() for p in self.powers]
        return torch.stack(xs).amax(0).mean(-1) - torch.stack(xs).amin(0).mean(-1)

    def x0(self): return (-3, 1.6)
    def domain(self): return (-10/3, 10/3), (-10/3, 10/3)
    def minima(self): return self.add
powdiff = PowDiff().register('powdiff')


class Eggholder(TestFunction):
    def __init__(self):
        super().__init__()


    def objective(self, X):
        x1, x2 = X
        return (-(x2 + 47) * torch.sin((x2 + x1/2 + 47).abs().sqrt()) - x1 * torch.sin((x1 - (x2 + 47)).abs().sqrt())) + 959.6407
    def x0(self): return (0, 0)
    def domain(self): return (-512, 512), (-512, 512)
    def minima(self): return (512, 404.2319)
eggholder = Eggholder().register('eggholder')



class PotentialWell(TestFunction):
    def __init__(self, add = (1, -2)):
        super().__init__()
        self.add = totensor(add)

    def objective(self, X):
        if X.ndim > 1:
            X = X.swapaxes(0, -1) # needed for broadcasting
            X = X + self.add
            X = X.swapaxes(0, -1)
        else:
            X = X + self.add

        x, y = X

        return (x**2 + y**2) * (1 + 0.5 * torch.sin(10 * x) * torch.sin(10 * y)) + 10 * torch.relu(x + y - 2)

    def x0(self): return (1, 3.3)
    def domain(self): return (-4, 2), (-1, 5)
    def minima(self): return None
potential_well = PotentialWell().register('potential_well')


class DoubleWell(TestFunction):
    def __init__(self,a=1.5, b=0.5):
        super().__init__()
        self.a = a; self.b = b

    def objective(self, X):
        x, y = X
        return (x**2 - self.a**2)**2 + 0.5*x*y + (y**2 - self.b**2)**2

    def x0(self): return (-0.01, 0.01)
    def domain(self): return (-2.5, 2.5), (-2.5, 2.5)
    def minima(self): return None
double_well = DoubleWell().register('double_well')



class DipoleField(TestFunction):
    """Magnetic Dipole Interaction Field"""
    def __init__(self):
        super().__init__()
    def objective(self, X):
        x, y = X
        eps = 1e-3
        term1 = -(x-1)/(((x-1)**2 + y**2 + eps)**1.5)
        term2 = -(x+1)/(((x+1)**2 + y**2 + eps)**1.5)
        return term1 + term2 + 0.1*(x**2 + y**2)


    def x0(self): return (0., 1.)
    def domain(self): return (-2, 2), (-2, 2)
    def minima(self): return None
dipole_field = DipoleField().register('dipole_field', 'dipole')




class ChaoticPotential(TestFunction):
    def __init__(self):
        super().__init__()
    def objective(self, X):
        x, y = X
        term1 = torch.sin(3*x) * torch.cos(4*y) * torch.exp(-0.1*(x**2 + y**2))
        term2 = 2*torch.abs(torch.sin(2*x) + torch.cos(3*y))
        term3 = 0.5*torch.relu(x**2 - y**2 - 1)
        return term1 + term2 + term3


    def x0(self): return (-3.3, 3.)
    def domain(self): return (-4, 4), (-4, 4)
    def minima(self): return None
chaotic_potential = ChaoticPotential().register('chaotic_potential')


class PlasmaSurface(TestFunction):
    def __init__(self):
        super().__init__()
    def objective(self, X):
        x, y = X
        return (x**4 + y**4) - 3*(x**2 + y**2) + 2*torch.abs(x*y) + 0.5*torch.sin(8*x)*torch.sin(8*y)


    def x0(self): return (-0.1, 0.1)
    def domain(self): return (-2, 2), (-2, 2)
    def minima(self): return None
plasma_surface = PlasmaSurface().register('plasma_surface')


class QuantumTunneling(TestFunction):
    def __init__(self):
        super().__init__()

    def objective(self, X):
        x, y = X
        base = (x**2 - 1)**2 + (y**2 - 1)**2
        noise = 0.2*torch.sin(15*x)*torch.sin(15*y)
        barrier = 2*torch.relu(x**2 + y**2 - 0.5)
        return base + noise + barrier

    def x0(self): return (-1.2, 2.)
    def domain(self): return (-2, 2), (-2, 2)
    def minima(self): return None
quantum_tunneling = QuantumTunneling().register('quantum_tunneling')


class ChemicalPotential(TestFunction):
    def __init__(self):
        super().__init__()

    def objective(self, X):
        x, y = X
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
    def domain(self): return (-1.5, 1.0), (-0.5, 2.0)
    def minima(self): return None
chemical_potential = ChemicalPotential().register('chemical_potential')



class FitnessLandscape(TestFunction):
    def __init__(self):
        super().__init__()

    def objective(self, X):
        x, y = X
        return (x**2 - 4*x + y**2 - 6*y +
                torch.cos(3*x) + torch.sin(2*y) +
                0.5*x*y + 13)

    def x0(self): return (4, 0.)
    def domain(self): return (-2, 5), (-2, 7)
    def minima(self): return None
fitness_landscape = FitnessLandscape().register('fitness_landscape', 'fitness')



class ArmError(TestFunction):
    """Minimize positioning error for a 2-link robotic arm reaching a target."""
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = 2.0, 1.0  # Link lengths
        self.target = torch.tensor([2.0, 1.0])  # Target position

    def objective(self, X):
        theta1, theta2 = X
        x = self.l1 * torch.cos(theta1) + self.l2 * torch.cos(theta1 + theta2)
        y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta1 + theta2)
        return (x - self.target[0])**2 + (y - self.target[1])**2

    def x0(self): return (-2, 0.)
    def domain(self): return (-torch.pi, torch.pi), (-torch.pi, torch.pi)
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

    def objective(self, X):
        x,y = X
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
    def domain(self): return (-1, 1), (-1, 1)
    def minima(self): return None

spiral = Spiral().register('spiral')
