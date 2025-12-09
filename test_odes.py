"""
Test the shooting method with multiple ODEs that have known analytical solutions.

This script defines several test problems and compares the numerical solution
from the shooting method against the exact analytical solution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from shooting_method import (
    ShootingConfig, shoot, integrate_ivp, to_first_order,
    plot_trajectories, plot_residuals, plot_timing
)


# ---------------------------------------------------------------------------
# Test Problem Definitions
# ---------------------------------------------------------------------------

def test_problem_1():
    """
    Problem 1: y'' = -π²y + π²x
    Boundary conditions: y(0) = 0, y(1) = 1
    Exact solution: y(x) = x
    """
    def ode(x, y, y_prime):
        return -(np.pi**2) * y + (np.pi**2) * x

    def exact_solution(x):
        return x

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, 1.0),
        boundary_values=(0.0, 1.0),
        initial_slope=0.5,
        slope_bracket=(0.0, 2.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Linear: y'' = -π²y + π²x"


def test_problem_2():
    """
    Problem 2: y'' = -4y
    Boundary conditions: y(0) = 0, y(π/2) = 1
    Exact solution: y(x) = sin(2x) / sin(π)
    Actually, let me use simpler BCs: y(0) = 0, y(π/4) = 1/√2
    Exact: y(x) = sin(2x)
    """
    def ode(x, y, y_prime):
        return -4 * y

    def exact_solution(x):
        # y(x) = sin(2x) satisfies y'' = -4y
        # y(0) = 0, y(π/4) = sin(π/2) = 1
        return np.sin(2 * x)

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, np.pi / 4),
        boundary_values=(0.0, 1.0),
        initial_slope=1.5,
        slope_bracket=(0.0, 3.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Harmonic: y'' = -4y"


def test_problem_3():
    """
    Problem 3: y'' = 2
    Boundary conditions: y(0) = 0, y(1) = 0
    Exact solution: y(x) = x(1-x) = x - x²
    """
    def ode(x, y, y_prime):
        return 2.0

    def exact_solution(x):
        return x - x**2

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, 1.0),
        boundary_values=(0.0, 0.0),
        initial_slope=0.5,
        slope_bracket=(0.0, 2.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Constant acceleration: y'' = 2"


def test_problem_4():
    """
    Problem 4: y'' = y
    Boundary conditions: y(0) = 1, y(1) = (e + 1/e) / 2
    Exact solution: y(x) = cosh(x) = (e^x + e^(-x)) / 2
    """
    def ode(x, y, y_prime):
        return y

    def exact_solution(x):
        return np.cosh(x)

    b = 1.0
    beta = np.cosh(b)

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, b),
        boundary_values=(1.0, beta),
        initial_slope=0.5,
        slope_bracket=(0.0, 2.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Exponential growth: y'' = y"


def test_problem_5():
    """
    Problem 5: y'' + y' - 2y = 0
    Boundary conditions: y(0) = 1, y(1) = 0
    Exact solution: y(x) = (e^(-2) - 1)/(e^(-2) - e) * e^x + (1 - e)/(e^(-2) - e) * e^(-2x)

    This is a second-order linear ODE with constant coefficients.
    Characteristic equation: r² + r - 2 = 0 → (r+2)(r-1) = 0 → r = -2, 1
    General solution: y = C₁e^x + C₂e^(-2x)
    """
    def ode(x, y, y_prime):
        return 2 * y - y_prime

    # Solving for C₁ and C₂ with y(0) = 1, y(1) = 0
    # y(0) = C₁ + C₂ = 1
    # y(1) = C₁*e + C₂*e^(-2) = 0
    # From second: C₁*e = -C₂*e^(-2), so C₁ = -C₂*e^(-3)
    # Substitute: -C₂*e^(-3) + C₂ = 1, so C₂(1 - e^(-3)) = 1
    C2 = 1.0 / (1.0 - np.exp(-3))
    C1 = 1.0 - C2

    def exact_solution(x):
        return C1 * np.exp(x) + C2 * np.exp(-2 * x)

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, 1.0),
        boundary_values=(1.0, 0.0),
        initial_slope=-0.5,
        slope_bracket=(-3.0, 1.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Damped: y'' + y' - 2y = 0"


def test_problem_6():
    """
    Problem 6: Nonlinear problem y'' = -y²
    Boundary conditions: y(0) = 1, y(1) ≈ 0.5
    This doesn't have a simple closed-form solution, but we can verify convergence.
    """
    def ode(x, y, y_prime):
        return -y**2

    # No exact solution available for this nonlinear problem
    exact_solution = None

    config = ShootingConfig(
        ode=ode,
        interval=(0.0, 1.0),
        boundary_values=(1.0, 0.5),
        initial_slope=-0.5,
        slope_bracket=(-2.0, 0.0),
        ivp_method="rk4",
        root_method="secant",
        grid_points=200,
        residual_tol=1e-8,
        max_root_iterations=25,
    )

    return config, exact_solution, "Nonlinear: y'' = -y²"


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

def run_test(problem_num, config, exact_solution, description):
    """Run a single test problem and return results."""
    print(f"\n{'='*70}")
    print(f"Test Problem {problem_num}: {description}")
    print(f"{'='*70}")
    print(f"ODE interval: [{config.interval[0]}, {config.interval[1]}]")
    print(f"Boundary values: y({config.interval[0]}) = {config.boundary_values[0]}, "
          f"y({config.interval[1]}) = {config.boundary_values[1]}")
    print(f"IVP method: {config.ivp_method}, Root finder: {config.root_method}")

    # Run shooting method
    result = shoot(config)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {len(result.records)}")
    print(f"Identified initial slope y'(a): {result.slope_star:.10f}")

    if result.records:
        final_residual = abs(result.records[-1].residual)
        print(f"Final residual |y(b) - β|: {final_residual:.3e}")

    # Get final solution
    system = to_first_order(config.ode)
    x_vals, y_vals, _ = integrate_ivp(
        system=system,
        interval=config.interval,
        y0=config.boundary_values[0],
        v0=result.slope_star,
        steps=config.grid_points,
        method=config.ivp_method,
    )

    # Compare with exact solution if available
    if exact_solution is not None:
        y_exact = exact_solution(x_vals)
        max_error = np.max(np.abs(y_vals - y_exact))
        l2_error = np.sqrt(np.mean((y_vals - y_exact)**2))
        print(f"\nError Analysis:")
        print(f"  Max error: {max_error:.3e}")
        print(f"  L2 error:  {l2_error:.3e}")

        # Check if exact initial slope matches
        if len(y_exact) > 1:
            h = x_vals[1] - x_vals[0]
            exact_slope = (y_exact[1] - y_exact[0]) / h  # Forward difference approximation
            print(f"  Exact y'(a) ≈ {exact_slope:.10f}")
            print(f"  Slope error: {abs(result.slope_star - exact_slope):.3e}")
    else:
        y_exact = None
        print(f"\nNo exact solution available (nonlinear problem)")

    return {
        'x': x_vals,
        'y_numerical': y_vals,
        'y_exact': y_exact,
        'result': result,
        'description': description,
        'config': config
    }


def plot_combined_trajectories_vectorfield(problem_num, data, filename):
    """Combine vector field and iteration trajectories in a single plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ode_func = data['config'].ode
    interval = data['config'].interval
    boundary_values = data['config'].boundary_values
    records = data['result'].records
    a, b = interval
    alpha, beta = boundary_values

    # Determine y-axis range from all iterations
    all_y_vals = [rec.y for rec in records]
    y_min = min(alpha, beta, min([y.min() for y in all_y_vals])) - 0.5
    y_max = max(alpha, beta, max([y.max() for y in all_y_vals])) + 0.5

    # Create grid for vector field
    x_range = np.linspace(a, b, 20)
    y_range = np.linspace(y_min, y_max, 20)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate direction field for the ODE
    U = np.ones_like(X)  # dx component
    V = np.zeros_like(Y)  # dy component

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            y_val = Y[i, j]
            try:
                v_approx = 0.0  # Approximate y' as 0 for direction field
                y_prime_prime = ode_func(x_val, y_val, v_approx)
                V[i, j] = y_prime_prime * 0.05  # Scale for visualization
            except:
                V[i, j] = 0

    # Normalize vectors for visualization
    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1
    U_norm = U / M
    V_norm = V / M

    # Plot vector field as background
    ax.quiver(X, Y, U_norm, V_norm, M, alpha=0.3, cmap='gray', scale=25, width=0.002, zorder=1)

    # Plot all iteration trajectories
    n_iters = len(records)
    for i, rec in enumerate(records):
        # Fade older iterations, make final iteration most visible
        alpha_val = 0.3 + 0.7 * (i / max(1, n_iters - 1))
        linewidth = 1.0 + 1.5 * (i / max(1, n_iters - 1))
        color = plt.cm.viridis(i / max(1, n_iters - 1))

        ax.plot(rec.x, rec.y, color=color, alpha=alpha_val, linewidth=linewidth,
                label=f"Iter {i}: slope={rec.slope:.4f}", zorder=3 + i)

    # Plot exact solution if available
    if data['y_exact'] is not None:
        ax.plot(data['x'], data['y_exact'], 'r--', linewidth=2.5,
                label='Exact Solution', alpha=0.8, zorder=10)

    # Plot boundary conditions
    ax.plot(a, alpha, 'ro', markersize=15, label=f'BC: y({a:.2f})={alpha:.2f}',
            zorder=20, markeredgecolor='darkred', markeredgewidth=2.5)
    ax.plot(b, beta, 'rs', markersize=15, label=f'BC: y({b:.2f})={beta:.2f}',
            zorder=20, markeredgecolor='darkred', markeredgewidth=2.5)

    # Add convergence info box
    converged = data['result'].converged
    iters = len(records)
    textstr = f"Converged: {converged}\nIterations: {iters}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, zorder=25)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Problem {problem_num}: {data["description"]}\n' +
                 'Shooting Iterations Over Vector Field',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved combined plot: {filename}")
    plt.close()


def main():
    """Run all test problems."""
    print("="*70)
    print("SHOOTING METHOD TEST SUITE")
    print("="*70)

    # Define all test problems
    test_problems = [
        test_problem_1(),
        test_problem_2(),
        test_problem_3(),
        test_problem_4(),
        test_problem_5(),
        test_problem_6(),
    ]

    # Run all tests
    results = {}
    for i, (config, exact_sol, description) in enumerate(test_problems, 1):
        results[i] = run_test(i, config, exact_sol, description)

    # Create plots directory
    os.makedirs("plots", exist_ok=True)

    # Generate individual plots for each test problem
    for problem_num, data in results.items():
        records = data['result'].records
        desc = data['description'].replace(':', '').replace(' ', '_')

        # Combined trajectory iterations + vector field plot
        plot_combined_trajectories_vectorfield(problem_num, data,
                                              f"plots/test_{problem_num}_{desc}_combined.png")

        # Diagnostic plots
        plot_residuals(records, f"plots/test_{problem_num}_{desc}_residuals.png")
        plot_timing(records, f"plots/test_{problem_num}_{desc}_timing.png")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total = len(results)
    converged = sum(1 for r in results.values() if r['result'].converged)
    print(f"Total problems: {total}")
    print(f"Converged: {converged}/{total}")

    # Print accuracy summary
    print(f"\nAccuracy Summary (for problems with exact solutions):")
    print(f"{'Problem':<10} {'Description':<30} {'Max Error':<15} {'L2 Error':<15}")
    print("-" * 70)
    for i, data in results.items():
        if data['y_exact'] is not None:
            max_err = np.max(np.abs(data['y_numerical'] - data['y_exact']))
            l2_err = np.sqrt(np.mean((data['y_numerical'] - data['y_exact'])**2))
            desc = data['description'][:28]
            print(f"{i:<10} {desc:<30} {max_err:<15.3e} {l2_err:<15.3e}")


if __name__ == "__main__":
    main()
