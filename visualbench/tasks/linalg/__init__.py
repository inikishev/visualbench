from .inverse_problems import InverseInverse, NewtonSchulzInverse, SinkhornInverse
from .inverses import Inverse, MoorePenrose
from .matrix_decompositions import (
    LU,
    QR,
    SVD,
    Bruhat,
    Cholesky,
    Eigen,
    Interpolative,
    LUPivot,
    Polar,
    CUR,
    NMF,
)
from .matrix_operators import MatrixLogarithm, MatrixRoot, MatrixSign
from .other import PCA, QEP, JordanForm, Whitening, LatticeBasisReduction, WahbaProblem, UnbalancedProcrustes
from .tensor_decompositions import (
    MPS,
    CanonicalPolyadic,
    CompactHOSVD,
    TensorTrain,
)
