import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# Main Manifold
def manifold_z(x, y, k=1.0):
    return np.sin(k * x) * np.cos(k * y)


def manifold_gradient(x, y, k=1.0):
    gx = k * np.cos(k * x) * np.cos(k * y)
    gy = -k * np.sin(k * x) * np.sin(k * y)
    return gx, gy


def manifold_hessian(x, y, k=1.0):
    fxx = -k**2 * np.sin(k * x) * np.cos(k * y)
    fyy = -k**2 * np.sin(k * x) * np.cos(k * y)
    fxy = -k**2 * np.cos(k * x) * np.sin(k * y)
    return np.array([[fxx, fxy], [fxy, fyy]])


def gaussian_curvature(x, y, k=1.0):
    H = manifold_hessian(x, y, k)
    return H[0, 0] * H[1, 1] - H[0, 1] ** 2


def mean_curvature(x, y, k=1.0):
    H = manifold_hessian(x, y, k)
    return (H[0, 0] + H[1, 1]) / 2.0


# Paths
def geodesic_path(ax, ay, bx, by, steps=120, k=1.0):
    ts = np.linspace(0, 1, steps)
    xs = ax + (bx - ax) * ts
    ys = ay + (by - ay) * ts
    zs = manifold_z(xs, ys, k)
    return xs, ys, zs


def linear_path(ax, ay, bx, by, steps=120, k=1.0):
    ts = np.linspace(0, 1, steps)
    xs = ax + (bx - ax) * ts
    ys = ay + (by - ay) * ts
    az = manifold_z(ax, ay, k)
    bz = manifold_z(bx, by, k)
    zs_lin = az + (bz - az) * ts     
    zs_man = manifold_z(xs, ys, k)        
    return xs, ys, zs_lin, zs_man


def geodesic_length(ax, ay, bx, by, steps=400, k=1.0):
    xs, ys, zs = geodesic_path(ax, ay, bx, by, steps, k)
    dx = np.diff(xs); dy = np.diff(ys); dz = np.diff(zs)
    return float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))


def euclidean_length(ax, ay, bx, by, k=1.0):
    az = manifold_z(ax, ay, k)
    bz = manifold_z(bx, by, k)
    return float(np.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2))


# Adversarial Attack
def adversarial_perturbation(x, y, epsilon=0.1, k=1.0):
    gx, gy = manifold_gradient(x, y, k)
    x_new = x + epsilon * gx
    y_new = y + epsilon * gy
    return x_new, y_new


def apply_shield_grid(xx, yy, epsilon=0.1, iters=5, k=1.0):
    xx_p, yy_p = xx.copy(), yy.copy()
    for _ in range(iters):
        gx, gy = manifold_gradient(xx_p, yy_p, k)
        norm = np.sqrt(gx**2 + gy**2) + 1e-8
        xx_p = xx_p + epsilon * gx / norm
        yy_p = yy_p + epsilon * gy / norm
    return xx_p, yy_p


# Surface for Plotting
def make_surface(x_range=(-3, 3), y_range=(-3, 3), n=60, k=1.0):
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    XX, YY = np.meshgrid(x, y)
    ZZ = manifold_z(XX, YY, k)
    return XX, YY, ZZ


def curvature_map(x_range=(-3, 3), y_range=(-3, 3), n=50, k=1.0):
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    XX, YY = np.meshgrid(x, y)
    KK = gaussian_curvature(XX, YY, k)
    return XX, YY, KK