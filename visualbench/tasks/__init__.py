from .image_rectanges import RectangleReconstructor
from .linalg import (
    LU,
    MPS,
    PCA,
    QEP,
    QR,
    SVD,
    Bruhat,
    CanonicalPolyadicDecomposition,
    Cholesky,
    CompactHOSVD,
    EigenDecomposition,
    InterpolativeDecomposition,
    Inverse,
    JordanForm,
    LUPivot,
    MatrixLogarithm,
    MatrixRoot,
    MatrixSign,
    # MatrixSqrt,
    MoorePenrose,
    TensorTrainDecomposition,
    Whitening,
)
from .operations import Sorting
from .packing import BoxPacking, RotatingBoxPacking, SpherePacking
from .synthetic import (
    AlphaBeta1,
    Convex,
    NonlinearMatrixFactorization,
    Rosenbrock,
    SelfRecurrent,
    Sphere,
)
