"""
Shooting method playground with multiple IVP integrators and root-finders.

How to use:
1. Edit the `define_problem()` function in the USER INPUT BLOCK to encode
   your second-order ODE y'' = g(x, y, y'), boundary values, interval, and
   preferred methods (RK4 / euler / backward_euler with Newton per step;
   root-finders: newton / secant / bisection).
2. Run `python shooting_method.py`. Figures are written to the plots/ folder:
   - plots/trajectories.png : solutions computed per shooting iteration
   - plots/residuals.png    : |y(b) - beta| per iteration
   - plots/timing.png       : wall time per iteration (root-finder step)

This file is intentionally verbose and opinionated so each moving part is clear.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np

# Use a headless backend so the script works in non-GUI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)


# ---------------------------------------------------------------------------
# USER INPUT BLOCK
# ---------------------------------------------------------------------------
def define_problem() -> "ShootingConfig":
    """
    Edit this function to describe your boundary-value problem.

    The ODE must be supplied in the form y'' = g(x, y, y').
    Boundaries: y(a) = alpha, y(b) = beta over the interval [a, b].

    You can switch integrators (`ivp_method`) and root-finders (`root_method`)
    without touching the rest of the script.
    """

    # Problem: y'' = -pi^2 y - pi^2 x, with y(0) = 0, y(1) = 1.
    # Exact solution: y(x) = x, so y'(0) = 1.
    def g(x: float, y: float, y_prime: float) -> float:
        return -(np.pi**2) * y + (np.pi**2) * x

    interval = (0.0, 1.0)
    boundary_values = (0.0, 1.0)

    # Initial guess for the unknown initial slope y'(a).
    slope_guess = 0.5

    # Optional bracket for methods that need it (bisection, secant).
    # Provide a tuple (slope_low, slope_high) such that the residuals
    # at those slopes have opposite signs.
    slope_bracket = (0.0, 2.0)

    # Choose IVP integrator: "rk4", "euler", or "backward_euler".
    ivp_method = "rk4"

    # Choose root-finder: "newton", "secant", or "bisection".
    # Use secant here because the finite-difference Newton step can be touchy
    # on this linear problem when starting far from the true slope.
    root_method = "secant"

    return ShootingConfig(
        ode=g,
        interval=interval,
        boundary_values=boundary_values,
        initial_slope=slope_guess,
        slope_bracket=slope_bracket,
        ivp_method=ivp_method,
        root_method=root_method,
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
        fd_step=1e-6,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
Vector = np.ndarray


@dataclass
class ShootingConfig:
    ode: Callable[[float, float, float], float]
    interval: Tuple[float, float]  # (a, b)
    boundary_values: Tuple[float, float]  # (alpha, beta)
    initial_slope: float
    slope_bracket: Optional[Tuple[float, float]] = None
    ivp_method: str = "rk4"
    root_method: str = "newton"
    grid_points: int = 200
    residual_tol: float = 1e-8
    max_root_iterations: int = 25
    fd_step: float = 1e-6  # finite-difference step for Newton/derivatives


@dataclass
class IterationRecord:
    slope: float
    residual: float
    elapsed: float
    x: Vector
    y: Vector


@dataclass
class ShootingResult:
    slope_star: float
    records: List[IterationRecord]
    converged: bool
    method: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def to_first_order(ode: Callable[[float, float, float], float]) -> Callable[[float, Vector], Vector]:
    """
    Convert y'' = g(x, y, y') into the first-order system:
    z = [y, v], z' = [v, g(x, y, v)].
    """

    def system(x: float, z: Vector) -> Vector:
        y, v = z
        return np.array([v, ode(x, y, v)], dtype=float)

    return system


def numerical_jacobian(
    f: Callable[[float, Vector], Vector], x: float, z: Vector, eps: float = 1e-6
) -> np.ndarray:
    """Finite-difference Jacobian of f wrt z using central differences."""
    n = len(z)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        z_plus = z.copy()
        z_plus[i] += eps
        z_minus = z.copy()
        z_minus[i] -= eps
        J[:, i] = (f(x, z_plus) - f(x, z_minus)) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# IVP integrators
# ---------------------------------------------------------------------------
def integrate_ivp(
    system: Callable[[float, Vector], Vector],
    interval: Tuple[float, float],
    y0: float,
    v0: float,
    steps: int,
    method: str = "rk4",
    jacobian: Optional[Callable[[float, Vector], np.ndarray]] = None,
) -> Tuple[Vector, Vector, Vector]:
    """
    Integrate z' = system(x, z) over [a, b] starting from z(a) = [y0, v0].
    Returns (x_grid, y_values, v_values).
    """
    a, b = interval
    h = (b - a) / steps
    x_grid = np.linspace(a, b, steps + 1)
    z = np.zeros((steps + 1, 2), dtype=float)
    z[0, :] = [y0, v0]

    if method.lower() == "euler":
        stepper = _step_euler
    elif method.lower() == "rk4":
        stepper = _step_rk4
    elif method.lower() == "backward_euler":
        stepper = _step_backward_euler
    else:
        raise ValueError(f"Unknown IVP method '{method}'. Use rk4, euler, or backward_euler.")

    for k in range(steps):
        z[k + 1] = stepper(
            system=system,
            x_k=x_grid[k],
            z_k=z[k],
            h=h,
            jacobian=jacobian,
        )

    return x_grid, z[:, 0], z[:, 1]


def _step_euler(
    system: Callable[[float, Vector], Vector],
    x_k: float,
    z_k: Vector,
    h: float,
    **_: dict,
) -> Vector:
    """Forward Euler step."""
    return z_k + h * system(x_k, z_k)


def _step_rk4(
    system: Callable[[float, Vector], Vector],
    x_k: float,
    z_k: Vector,
    h: float,
    **_: dict,
) -> Vector:
    """Classic 4th-order Runge-Kutta step."""
    k1 = system(x_k, z_k)
    k2 = system(x_k + 0.5 * h, z_k + 0.5 * h * k1)
    k3 = system(x_k + 0.5 * h, z_k + 0.5 * h * k2)
    k4 = system(x_k + h, z_k + h * k3)
    return z_k + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _step_backward_euler(
    system: Callable[[float, Vector], Vector],
    x_k: float,
    z_k: Vector,
    h: float,
    jacobian: Optional[Callable[[float, Vector], np.ndarray]] = None,
    newton_tol: float = 1e-10,
    newton_maxiter: int = 12,
    **_: dict,
) -> Vector:
    """
    Backward Euler step solved with Newton's method:
    z_{k+1} = z_k + h * f(x_{k+1}, z_{k+1}).
    """
    x_next = x_k + h
    z_guess = z_k.copy()
    jac_fn = jacobian or (lambda x, z: numerical_jacobian(system, x, z))

    for _ in range(newton_maxiter):
        F = z_guess - z_k - h * system(x_next, z_guess)
        if np.linalg.norm(F, ord=2) < newton_tol:
            break
        J = np.eye(len(z_k)) - h * jac_fn(x_next, z_guess)
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian became singular during backward Euler step.")
        z_guess = z_guess + delta

    return z_guess


# ---------------------------------------------------------------------------
# Shooting (root finding in the initial slope)
# ---------------------------------------------------------------------------
def shoot(
    config: ShootingConfig,
) -> ShootingResult:
    """
    Run the shooting method using the chosen IVP integrator and root-finder.
    Returns the identified slope and a record of each iteration.
    """
    system = to_first_order(config.ode)
    alpha, beta = config.boundary_values

    def residual_for_slope(slope: float) -> Tuple[float, IterationRecord]:
        """Compute residual y(b) - beta for a given trial slope."""
        start = time.perf_counter()
        x, y, v = integrate_ivp(
            system=system,
            interval=config.interval,
            y0=alpha,
            v0=slope,
            steps=config.grid_points,
            method=config.ivp_method,
        )
        elapsed = time.perf_counter() - start
        residual = y[-1] - beta
        record = IterationRecord(slope=slope, residual=residual, elapsed=elapsed, x=x, y=y)
        return residual, record

    if config.root_method.lower() == "newton":
        result = _root_newton(
            residual_for_slope,
            slope0=config.initial_slope,
            tol=config.residual_tol,
            maxiter=config.max_root_iterations,
            fd_step=config.fd_step,
        )
    elif config.root_method.lower() == "secant":
        if not config.slope_bracket:
            raise ValueError("Secant method requires slope_bracket = (s0, s1).")
        s0, s1 = config.slope_bracket
        result = _root_secant(
            residual_for_slope,
            s0=s0,
            s1=s1,
            tol=config.residual_tol,
            maxiter=config.max_root_iterations,
        )
    elif config.root_method.lower() == "bisection":
        if not config.slope_bracket:
            raise ValueError("Bisection requires slope_bracket = (s_low, s_high).")
        s_low, s_high = config.slope_bracket
        result = _root_bisection(
            residual_for_slope,
            a=s_low,
            b=s_high,
            tol=config.residual_tol,
            maxiter=config.max_root_iterations,
        )
    else:
        raise ValueError(f"Unknown root-finder '{config.root_method}'. Use newton, secant, or bisection.")

    return ShootingResult(
        slope_star=result["slope"],
        records=result["records"],
        converged=result["converged"],
        method=config.root_method,
    )


def _root_newton(
    residual_fn: Callable[[float], Tuple[float, IterationRecord]],
    slope0: float,
    tol: float,
    maxiter: int,
    fd_step: float,
) -> dict:
    records: List[IterationRecord] = []
    slope = slope0
    converged = False

    for _ in range(maxiter):
        res, rec = residual_fn(slope)
        extra_time = 0.0
        # Finite-difference derivative of residual wrt slope.
        res_plus, rec_plus = residual_fn(slope + fd_step)
        res_minus, rec_minus = residual_fn(slope - fd_step)
        extra_time += rec_plus.elapsed + rec_minus.elapsed
        rec.elapsed += extra_time
        records.append(rec)
        if abs(res) < tol:
            converged = True
            break

        deriv = (res_plus - res_minus) / (2 * fd_step)
        if deriv == 0:
            raise RuntimeError("Newton derivative vanished; try secant or bisection.")
        slope = slope - res / deriv

    return {"slope": slope, "records": records, "converged": converged}


def _root_secant(
    residual_fn: Callable[[float], Tuple[float, IterationRecord]],
    s0: float,
    s1: float,
    tol: float,
    maxiter: int,
) -> dict:
    records: List[IterationRecord] = []
    r0, rec0 = residual_fn(s0)
    records.append(rec0)
    r1, rec1 = residual_fn(s1)
    records.append(rec1)
    slope_prev, slope = s0, s1
    converged = False

    for _ in range(maxiter):
        if abs(r1 - r0) < 1e-14:
            raise RuntimeError("Secant method stagnated (residuals too close).")
        slope_next = slope - r1 * (slope - slope_prev) / (r1 - r0)

        r_next, rec_next = residual_fn(slope_next)
        records.append(rec_next)

        if abs(r_next) < tol:
            slope = slope_next
            converged = True
            break

        slope_prev, slope = slope, slope_next
        r0, r1 = r1, r_next

    return {"slope": slope, "records": records, "converged": converged}


def _root_bisection(
    residual_fn: Callable[[float], Tuple[float, IterationRecord]],
    a: float,
    b: float,
    tol: float,
    maxiter: int,
) -> dict:
    records: List[IterationRecord] = []
    r_left, rec_a = residual_fn(a)
    r_right, rec_b = residual_fn(b)
    records.extend([rec_a, rec_b])

    if r_left * r_right > 0:
        raise ValueError("Bisection requires residuals of opposite sign at the bracket ends.")

    converged = False
    left, right = a, b

    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        r_mid, rec_mid = residual_fn(mid)
        records.append(rec_mid)

        if abs(r_mid) < tol or 0.5 * (right - left) < tol:
            converged = True
            left = right = mid
            break

        if r_left * r_mid < 0:
            right = mid
            r_right = r_mid
        else:
            left = mid
            r_left = r_mid

    slope_est = 0.5 * (left + right)
    return {"slope": slope_est, "records": records, "converged": converged}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_trajectories(records: Iterable[IterationRecord], filename: str) -> None:
    plt.figure(figsize=(7, 4))
    for i, rec in enumerate(records):
        plt.plot(rec.x, rec.y, label=f"iter {i}: slope={rec.slope:.5f}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Trajectories per shooting iteration")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_residuals(records: Iterable[IterationRecord], filename: str) -> None:
    residuals = [abs(r.residual) for r in records]
    plt.figure(figsize=(6, 3.5))
    plt.semilogy(range(len(residuals)), residuals, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("|y(b) - beta|")
    plt.title("Residual magnitude")
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_timing(records: Iterable[IterationRecord], filename: str) -> None:
    times = [r.elapsed for r in records]
    plt.figure(figsize=(6, 3.5))
    plt.bar(range(len(times)), times, color="tab:orange")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Computation time per iteration")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def main() -> None:
    config = define_problem()
    result = shoot(config)

    print(f"Method combo: IVP={config.ivp_method}, root={config.root_method}")
    print(f"Converged: {result.converged}")
    print(f"Identified initial slope y'(a): {result.slope_star:.8f}")
    print(f"Iterations: {len(result.records)}")
    if result.records:
        print(f"Final residual |y(b)-beta|: {abs(result.records[-1].residual):.3e}")

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    plot_trajectories(result.records, filename="plots/trajectories.png")
    plot_residuals(result.records, filename="plots/residuals.png")
    plot_timing(result.records, filename="plots/timing.png")
    print("Saved plots: plots/trajectories.png, plots/residuals.png, plots/timing.png")


if __name__ == "__main__":
    main()
