# pylint:disable=no-member, not-callable
import cv2
import numpy as np
import torch

from ...benchmark import Benchmark


class CrouzeixConjecture(Benchmark):
    """optimize a ball"""
    def __init__(self, n=8, poly_degree=8):
        super().__init__()
        self.n = n
        self.poly_degree = poly_degree

        # A: The matrix to optimize (complex)
        # We initialize with a random complex matrix
        self.A_real = torch.nn.Parameter(torch.randn(n, n) * 0.1)
        self.A_imag = torch.nn.Parameter(torch.randn(n, n) * 0.1)

        # p: Polynomial coefficients (complex)
        self.c_real = torch.nn.Parameter(torch.randn(poly_degree + 1) * 0.1)
        self.c_imag = torch.nn.Parameter(torch.randn(poly_degree + 1) * 0.1)

        # Sampling points for the boundary of W(A)
        self.thetas = torch.nn.Buffer(torch.linspace(0, 2 * np.pi, 128))

    def get_A(self):
        return torch.complex(self.A_real, self.A_imag)

    def get_poly_coeffs(self):
        return torch.complex(self.c_real, self.c_imag)

    def eval_poly_mat(self, A, coeffs):
        # Horner's method for matrix polynomial evaluation
        res = torch.zeros_like(A)
        res = res + coeffs[-1] * torch.eye(self.n, device=A.device, dtype=A.dtype)
        for i in range(len(coeffs) - 2, -1, -1):
            res = torch.matmul(res, A) + coeffs[i] * torch.eye(self.n, device=A.device, dtype=A.dtype)
        return res

    def get_loss(self):
        A = self.get_A()
        coeffs = self.get_poly_coeffs()

        # 1. Compute ||p(A)|| (Spectral Norm)
        pA = self.eval_poly_mat(A, coeffs)
        # s_max is the matrix 2-norm
        s_max = torch.linalg.matrix_norm(pA, ord=2)

        # 2. Compute Max |p(z)| for z in W(A)
        # Boundary points of W(A) are found via the largest eigenvalue of (exp(-it)A + exp(it)A*)/2
        A_h = A.mH # Conjugate transpose

        # We find the boundary points z_theta
        # H_theta = (e^{-i theta} A + e^{i theta} A^*) / 2
        #exp_theta = torch.exp(-1j * self.thetas)
        z_boundary = []

        for t in self.thetas:
            H_t = (torch.exp(-1j * t) * A + torch.exp(1j * t) * A_h) / 2
            eigvals, eigvecs = torch.linalg.eigh(H_t)
            v = eigvecs[:, -1:] # Eigenvector for largest eigenvalue
            z = (v.mH @ A @ v).squeeze()
            z_boundary.append(z)

        z_boundary = torch.stack(z_boundary)

        # Evaluate polynomial at these boundary points
        # p(z) = sum c_k z^k
        pz_vals = torch.zeros_like(z_boundary, dtype=A.dtype)
        for i, c in enumerate(coeffs):
            pz_vals += c * (z_boundary ** i)

        max_pz = torch.max(torch.abs(pz_vals))

        # Crouzeix ratio: ratio = ||p(A)|| / max_{z in W(A)} |p(z)|
        # We want to maximize this ratio, so we minimize -ratio
        # Add a small epsilon to avoid division by zero
        ratio = s_max / (max_pz + 1e-8)

        self.log("ratio", ratio)
        self.log("||p(A)||", s_max)
        self.log("max|p(z)|", max_pz)

        if self._make_images:
            img = self.render_numerical_range(A, z_boundary, pz_vals)
            self.log_image(name="NumericalRange", image=img, to_uint8=False)

        # Minimize negative ratio
        return -ratio

    def render_numerical_range(self, A, z_boundary, pz_vals):
        # Fast visualization using OpenCV
        W, H = 512, 512
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Get eigenvalues for spectrum visualization
        eigs = torch.linalg.eigvals(A).detach().cpu().numpy()
        z_b = z_boundary.detach().cpu().numpy()

        # Scale coordinates to fit image
        all_pts = np.concatenate([eigs, z_b])
        min_x, max_x = all_pts.real.min() - 0.5, all_pts.real.max() + 0.5 # type:ignore
        min_y, max_y = all_pts.imag.min() - 0.5, all_pts.imag.max() + 0.5 # type:ignore

        def to_px(z):
            x = int((z.real - min_x) / (max_x - min_x) * W)
            y = int((z.imag - min_y) / (max_y - min_y) * H)
            return (x, H - y)

        # Draw Numerical Range Boundary
        pts = np.array([to_px(z) for z in z_b], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Fill Numerical Range with transparent green
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 100, 0))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw Eigenvalues
        for e in eigs:
            cv2.circle(canvas, to_px(e), 4, (0, 0, 255), -1)

        # Legend/Text
        cv2.putText(canvas, f"Matrix Size: {self.n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return canvas

