# <h1 align='center'>visualbench</h1>

Fast benchmarks for optimization algorithms - PyTorch optimizers as well as solvers from any other libraries such as scipy.optimize, optuna, etc.

Many benchmarks support visualization where you can plot or render a video to see how the solution is being optimized.

### Installation

```bash
pip install visualbench
```

The dependencies are `pytorch`, `numpy`, `scipy`, `matplotlib` and `opencv-python`.

Few benchmarks also use `torchvision`, `scikit-learn`, `mnist1d`, `gpytorch`.

### Function descent

Useful to debug optimizers:

```python
import torch
import visualbench as vb

# "booth" is a pre-defined function
# can also pass custom one like `lambda x, y: x**2 + y**2`
bench = vb.FunctionDescent("booth")
opt = torch.optim.Adam(bench.parameters(), 1e-1)

bench.run(opt, max_steps=1000)
```

we can now plot a visualization:

```python
bench.plot()
```

<img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/7c561126-c2ed-4476-ae5f-c1b374f0e9f3" />

or render it to MP4/GIF (I recommend MP4 because its much faster to render)

```python
bench.render("Adam.mp4")
```

<img width="400" height="auto" alt="image" src="https://github.com/inikishev/visualbench/blob/main/assets/readme/Adam.gif" />

### Colorization

Here is GD with momentum on the colorization problem from <https://distill.pub/2017/momentum/>. The objective is to minimize differences between adjacent pixels.

```python
bench = vb.Colorization().cuda()
opt = torch.optim.SGD(bench.parameters(), lr=2e-1, momentum=0.999)
bench.run(opt, 1000)
bench.render("Colorization.mp4")
```

<https://github.com/user-attachments/assets/37b32d75-6f80-4c6b-a360-254aea368aa8>

### NeuralDrawer

In this objective the goal is to train a neural network which predicts pixel color of a given image based on its two coordinates:

```python
import heavyball

bench = vb.NeuralDrawer(
    vb.data.WEEVIL96, # path to an image file, or a numpy array/torch tensor
    vb.models.MLP([2,16,16,16,16,16,16,16,3], bn=True), # neural net
    expand=48 # renders 48 pixels outside of the image
).cuda() # don't forget to move to CUDA!

opt = heavyball.ForeachSOAP(bench.parameters(), lr=1e-2)
bench.run(opt, 1000)
bench.render("NeuralDrawer.mp4", scale=2)
```

<https://github.com/user-attachments/assets/99031a4f-d2aa-4640-b940-dc87c3316fdb>

# <h1 align='center'>Problems</h1>

#### Linear algebra

So for example SVD decomposes A into USV*, where U and V are orthonormal unitary, S is diagonal. So in `SVD` benchmark we optimize U, S and V so that USV* approximates A, and so that U and V are orthonormal.

Stochastic versions usually work by using matrix-vector products with random vectors. For example in `StochasticMatrixRecovery` we optimize B to recover A by using loss = mse(Av, Bv) where v is a vector sampled randomly on each step. It also computes test loss as mse(A, B). Those are very fast to evaluate and might be good proxies for real stochastic ML tasks.

All of them:

`LDL, LU, LUP, NNMF, QR, SVD, Cholesky, Eigendecomposition, EigenWithInverse, KroneckerFactorization, Polar, RankFactorization, Drazin, Inverse, MoorePenrose, StochasticInverse, StochasticRLstsq, Preconditioner, StochasticPreconditioner, LeastSquares, MatrixIdempotent, MatrixLogarithm, MatrixRoot, StochasticMatrixIdempotent, StochasticMatrixRoot, StochasticMatrixSign, StochasticMatrixRecovery, BilinearLeastSquares, TensorRankDecomposition, TensorSpectralNorm`

#### Drawing

- `LinesDrawer` - optimize lines to reconstruct an image
- `PartitionDrawer` - optimize partitions to reconstruct an image
- `RectanglesDrawer` - optimize rectangles to reconstruct an image
- `NeuralDrawer` - neural net predicts pixel color based on its two coordinates
- `LayerwiseNeuralDrawer` - same as `NeuralDrawer` but it also visualizes all intermediate layers

#### 2D functions

You can pass a function like `lambda x,y: x**2 + y**2`, or string name of one of pre-defined functions of which there are many, I usually use `"booth"`, `"rosen"`, and `"ill"` which is a rotated ill-conditioned quadratic.

- `FunctionDescent` - to see how optimizer descends a 2D function.
- `NeuralDescent` - optimize a model to output coordinates that minimize a 2D function. This is a great way to test how much curvature an optimizer actually uses on larger models.
- `SimultaneousFunctionDescent` - same as FunctionDescent, except the optimizer optimizes all points at the same time.
- `MetaLearning` - the goal is to optimize hyperparameters of an optimizer to descend a 2D function.

#### Packing / Covering

`BoxPacking`, `RigidBoxPacking`, `SpherePacking`, `RigidBoxCovering`

#### Projected objectives

This projects the trajectory of an optimizer on some problem, like neural network training, to a 2D subspace, and shows how optimizer navigates the landscape. It's actually very hard to get a good projection that doesn't bounce around and that is at least somewhat informative. I ended up using orthogonalized subspace defined by best point so far, point 5% before and point 10% before, with smoothing. On multi-dimensional rosenbrock it looks good, but neural net is still too chaotic.

All of those are in `vb.projected` namespace e.g. `vb.projected.Rosenbrock`:

`Ackley`, `BumpyBowl`, `ChebushevRosenbrock`,
`NeuralNet`, `Rastrigin`, `Rosenbrock`, `RotatedQuadratic`,

#### Datasets

Training models on various datasets. Those benchmarks are basically as fast as they can be as datasets are pre-loaded into memory and use a custom very fast dataloader for mini-batching.

In all of them it says shape of input and output in the docstring. So you need to specify any model (`torch.nn.Module`) that receives and outputs those shapes, or use something from `vb.models`.

Datasets with two features (like `Moons`) support visualizing/rendering the decision boundary.

- sklearn datasets (requires `scikit-learn`): `CaliforniaHousing`, `Moons`, `OlivettiFaces`, `OlivettiFacesAutoencoding`, `Covertype`, `KDDCup1999`, `Digits`, `Friedman1`, `Friedman2`, `Friedman3`
- mnist1d (requires `mnist1d`): `Mnist1d`, `Mnist1dAutoencoding`
- Segmentation: `SynthSeg1d` (synthetic 1d semantic segmentation dataset)
- torchvision: `MNIST`, `FashionMNIST`, `FashionMNISTAutoencoding`, `EMNIST`, `CIFAR10`, `CIFAR100`
- other: `WDBC`

#### Other machine learning

- `TSNE` - T-SNE dimensionality reduction with visualization
- `Glimmer` - Glimmer dimensionality reduction with visualization
- `GaussianMixtureNLL` - optimize a gaussian mixture and visualizes PCA-projected decision boundaries
- `MFMovieLens` - matrix factorization on MovieLens dataset
- `WavePINN` - trains PINN on wave PDE
- `AffineRegistration`, `DeformableRegistration` - image registration (2D and 3D)
- `StyleTransfer` - VGG style transfer
- `GaussianProcesses` (reguires GPytorch) - reconstruct 2D function with GP

#### Synthetic problems

- `Sphere`, `Rosenbrock`, `ChebushevRosenbrock`, `RotatedQuadratic`, `Rastrigin`, `Ackley`.

#### Uncategorized problems

- `AlphaEvolveB1` - Alpha Evolve B1 problem (code from <https://github.com/damek/alpha_evolve_problem_B1/blob/main/problemb1.ipynb>), with visualization
- `MuonCoeffs` - optimize Muon coefficients, this is <https://leloykun.github.io/ponder/muon-opt-coeffs/> ported to pytorch
- `HumanHeartDipole`, `PropaneCombustion` - two real-world least squares problems from MINPACK2
- `Colorization` - colorization problem from <https://distill.pub/2017/momentum/>
- `GraphLayout` - optimize graph layout
- `OptimalControl` - optimize trajectory
- `CUTEst` - wrapper for CUTEst (requires `pycustest`), with a custom torch.autograd.Function function that wraps CUTEst's gradients and hessian-vector products.

# <h1 align='center'>Gallery</h1>

I have to make this repo public to enable github pages, so those links are temorarily empty!

- [More videos](wait)

- [How much curvature do second order optimizers actually use?](wait)

# <h1 align='center'>Advanced</h1>

### Custom training loops

Benchmarks have `closure` method which returns the loss and optionally computes the gradients. This way one can write a custom training loop:

```py
bench = vb.Mnist1d(
    vb.models.MLP([40, 64, 96, 128, 256, 10], act_cls=torch.nn.ELU),
    batch_size=32, test_batch_size=None,
).cuda()

# test every 10 forward passes
bench.set_test_intervals(test_every_forwards=10)

# disable printing
bench.set_print_inverval(None)


opt = torch.optim.AdamW(bench.parameters(), 3e-3)

for step in range(1000):
    opt.zero_grad()
    loss = bench.closure(backward=False)
    loss.backward()
    opt.step()


print(f'{loss = }')
bench.plot()
```

### Non-pytorch optimizers

Solvers from other libraries can also be benchmarked/visualized easily.

Many solvers work with numpy vectors, so we can get all parameters of a benchmark concatenated to a vector like this:

```python
x0 = bench.get_x0().numpy(force=True)
```

To evaluate benchmark at parameters given in vector `x`, use `fx = bench.loss_at(x)`, `fx` will be a float.

To evaluate loss and gradient, use `fx, gx = bench.loss_grad_at(x)`. Here `gx` is a numpy array of the same length as `x`.

Using this, we can easily run solvers from other frameworks, for example scipy.optimize:

```python
import scipy.optimize

bench = vb.NeuralDrawer(
    vb.data.WEEVIL96,
    vb.models.MLP([2,16,16,16,16,16,16,16,3], bn=True),
    expand=48
).cuda()

x0 = bench.get_x0().numpy(force=True)

sol = scipy.optimize.minimize(
    fun = bench.loss_grad_at, # or `bench.loss_at` if gradient-free
    x0 = bench.get_x0().numpy(force=True),
    method = 'l-bfgs-b',
    jac = True, # fun returns (fx, gx)
    options = {"maxiter": 1000}
)

bench.plot()
```

Here is how to visualize optuna's TPE sampler on rosenbrock function:

```python
import numpy as np
import optuna
optuna.logging.disable_default_handler() # hides very useful information

bench = vb.FunctionDescent('rosen')

sampler = optuna.samplers.TPESampler(prior_weight = 2.0,)
study = optuna.create_study(sampler=sampler)

x0 = bench.get_x0().numpy(force=True)

def objective(trial: optuna.Trial):
    values = [trial.suggest_float(f"p{i}", -3, 3) for i in range(len(x0))]
    return bench.loss_at(np.asarray(values))

study.optimize(objective, n_trials=1000)

bench.render("Optuna.mp4", line_alpha=0.1)
```

<https://github.com/user-attachments/assets/021846d8-626d-4f2a-a8cb-7d2143d28673>

### Algebras

Some benchmarks let you choose an algebra, i.e. tropical algebra so that you can optimize tropical SVD decomposition etc. In tropical algebra plus is replaced with min, and product with plus. Whenever a benchmark has `algebra` argument, you can choose a different algebra by passing one of those strings:

```py
'elementary', 'tropical', 'tropical min', 'tropical max', 'fuzzy', 'lukasiewicz', 'viterbi', 'viterbi max', 'viterbi min', 'log', 'probabilistic', 'modulo1', 'modulo5'
```

### Adding noise

It is possible to add two kinds of noise to any benchmark by calling `benchmark.set_noise` method. First kind of noise `p` evaluates function and gradient at randomly perturbed parameters. Second kind of noise `g` is just noise added to gradients.

```py
bench = vb.FunctionDescent("booth").set_noise(p=2.0, g=2.0)
opt = torch.optim.SGD(bench.parameters(), lr=1e-2)

bench.run(opt, max_steps=1000)
bench.plot()
```

<img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/9262f339-3fda-4ec5-beb6-0777ca5a3fdb" />

### Multi-objective / Least squares optimization

Some benchmarks support returning multiple objectives or residuals for least squares. By default they return a single scalar value (usually sum or sum of squares, the function has to be explicitly defined in the benchmark). So to make a benchmark return multiple values, call `benchmark.set_multiobjective(True)`. Now whenever `bench.closure` is called, it returns a vector of values.

```py
import torchzero as tz

# rosenmo is version of rosenbrock which returns two residuals.
bench = vb.FunctionDescent("rosenmo")

# don't forget to enable multi-objective mode
bench.set_multiobjective(True)

#  We can use a least squares solver such as Gauss-Newton
opt = tz.Optimizer(
    bench.parameters(),
    tz.m.LevenbergMarquardt(tz.m.GaussNewton())
)

bench.run(opt, max_steps=100)
bench.plot()
```

<img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/a3c9ae79-972a-42fc-8af4-6f850a7faf80" />

### Logger

Benchmark has a logger object where all the metrics reside. For example you can get a dictionary which maps step to train loss like this: `train_loss = bench.logger["train loss"]`.

### More tips

- don't forget to move benchmarks to CUDA! Most are much faster on CUDA than on CPU.

- whenever a benchmark accepts an image or a matrix, you can pass numpy array, torch tensor, or path to an image.

- if you don't need visualization, use `benchmark.set_performance_mode(True)` to disable it which makes some benchmarks much faster by not running visualization code.

- to disable the stupid printing use `benchmark.set_print_inverval(None)`.

- benchmarks have a `benchmark.reset()` method, which resets the benchmark to initial state. It can be much faster than re-creating the benchmark from scratch in some cases, so it is good for hyperparameter tuning.

- if you use optuna pruner, use `benchmark.set_optuna_trial(trial, metric="train loss")` and it will report that metric to optuna and raise `optuna.TrialPruned()` when requested. See the hyperparameter optimization with Optuna example

# Defining new benchmarks

`Benchmark` is a subclass of `torch.nn.Module`.

To make a benchmark, subclass `Benchmark` and define a `get_loss` method which returns a `torch.Tensor` loss.

You can log any metrics by `self.log(value)`, and log any images by `self.log_image(image)`. The images will automatically be added to the plots and videos.

Here is an example of objective where the task is to invert a matrix, which also visualizes current solution and some losses:

```py
class MatrixInverse(vb.Benchmark):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()

        # store matrix and eye as buffers, that way benchmark.cuda() will move them to cuda
        self.matrix = torch.nn.Buffer(matrix.float().cpu())
        self.eye = torch.nn.Buffer(torch.eye(matrix.size(0), dtype=torch.float32))

        # this will be optimized to approximate inverse of the matrix
        self.inverse_hat = torch.nn.Parameter(torch.randn_like(self.matrix))

        # this shows the matrix on the video
        self.add_reference_image(name="matrix", image=self.matrix, to_uint8=True)

    def get_loss(self):

        # compute loss, B is inverse of A if AB = BA = I
        AB = self.matrix @ self.inverse_hat
        BA = self.inverse_hat @ self.matrix

        loss1 = torch.nn.functional.mse_loss(AB, BA)
        loss2 = torch.nn.functional.mse_loss(AB, self.eye)
        loss3 = torch.nn.functional.mse_loss(BA, self.eye)

        # log individual losses
        # note that if metric name doesn't start with "train " or "test ",
        # it be inserted in front of the name (this is by design)
        self.log("AB BA loss", loss1)
        self.log("AB identity loss", loss2)
        self.log("BA identity loss", loss3)

        # log images
        # skip (possibly expensive) visualization ocde if performance mode is enabled
        # which sets `self._make_images=False`
        if self._make_images:

            # to_uint8 normalizes them to and converts to uint8 data type
            # or you can do that manually if you want but then logged images must be uin8
            self.log_image(name='inverse', image=self.inverse_hat, to_uint8=True)
            self.log_image(name='AB', image=AB, to_uint8=True)
            self.log_image(name='BA', image=BA, to_uint8=True)
            self.log_image(name="reconstruction", image=torch.linalg.inv(self.inverse_hat), to_uint8=True)

        return loss1 + loss2 + loss3


# Done! We can now run the benchmark
matrix = torch.randint(0, 3, (64, 64)).sort(dim=0)[0]
benchmark = MatrixInverse(matrix).cuda()
optimizer = torch.optim.LBFGS(benchmark.parameters(), line_search_fn='strong_wolfe')
benchmark.run(optimizer, max_passes=1000)

benchmark.plot(yscale="log") # plots everything that was logged
benchmark.render("L-BFGS inverting a matrix.mp4", scale=4) # renders a video with images that were logged
```

<https://github.com/user-attachments/assets/0c768529-2e71-4667-8908-86bbce515852>

# License

MIT
