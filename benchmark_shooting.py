"""
Benchmark suite for shooting-method BVPs.

This script encodes a collection of boundary-value problems, calls an external
`shooting_solve` implementation, and generates publication-ready plots.

Note: Relies on `shooting_solve` from shooting_method.py. Adjust the solver
kwargs in `DEFAULT_SOLVER_KWARGS` if your implementation uses different names.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from shooting_method import shooting_solve

plt.rcParams.update({"font.size": 12})

# Optional resource monitoring
try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:  # pragma: no cover - optional dependency
    import pynvml  # type: ignore

    pynvml.nvmlInit()
except Exception:  # pragma: no cover
    pynvml = None


# ---------------------------------------------------------------------------#
# Problem definition                                                         #
# ---------------------------------------------------------------------------#
ODEFunction = Callable[[float, float, float], float]
ExactSolution = Callable[[np.ndarray], np.ndarray]


@dataclass
class Problem:
    name: str
    ode_fun: ODEFunction
    a: float
    b: float
    alpha: float
    beta: float
    s0: float
    category: str
    exact_solution: Optional[ExactSolution] = None
    parameters: Optional[Dict[str, Any]] = None


# Exact solutions for selected problems
def exact_solution_1(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x)
    return 0.0 * x_arr


def exact_solution_2(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x)
    return np.sin(np.pi * x_arr)


def exact_solution_3(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x)
    return np.exp(x_arr)


def exact_solution_4(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x)
    a_val = np.sqrt(2.0)
    K = (np.exp(4.0) - np.cosh(2.0 * a_val)) / np.sinh(2.0 * a_val)
    return np.exp(-x_arr) * (np.cosh(a_val * x_arr) + K * np.sinh(a_val * x_arr))


def exact_solution_5(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x)
    return 2.0 * np.cos(2.0 * x_arr) / np.cos(2.0)


def exact_solution_11(x: np.ndarray | float, eps: float = 0.01) -> np.ndarray:
    x_arr = np.asarray(x)
    return (np.exp((1.0 - x_arr) / eps) - 1.0) / (np.exp(1.0 / eps) - 1.0)


def build_problems() -> List[Problem]:
    """Construct the full list of benchmark problems."""
    problems: List[Problem] = []

    # 1) Linear second-order ODE with trivial solution
    problems.append(
        Problem(
            name="trivial_linear",
            ode_fun=lambda x, y, yp: -(np.pi**2) * y,
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=0.0,
            s0=0.1,
            category="easy",
            exact_solution=exact_solution_1,
        )
    )

    # 2) Sine right-hand side
    problems.append(
        Problem(
            name="sine_rhs",
            ode_fun=lambda x, y, yp: -(np.pi**2) * np.sin(np.pi * x),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=0.0,
            s0=1.0,
            category="easy",
            exact_solution=exact_solution_2,
        )
    )

    # 3) Exponential growth
    problems.append(
        Problem(
            name="exponential_growth",
            ode_fun=lambda x, y, yp: y,
            a=0.0,
            b=1.0,
            alpha=1.0,
            beta=float(np.e),
            s0=1.0,
            category="easy",
            exact_solution=exact_solution_3,
        )
    )

    # 4) Linear ODE with first-derivative term: y'' + 2y' - y = 0
    r1 = -1.0 + np.sqrt(2.0)
    r2 = -1.0 - np.sqrt(2.0)
    mat = np.array([[1.0, 1.0], [np.exp(2 * r1), np.exp(2 * r2)]])
    rhs = np.array([1.0, np.exp(2.0)])
    c1, c2 = np.linalg.solve(mat, rhs)
    problems.append(
        Problem(
            name="damped_linear_with_first_derivative",
            ode_fun=lambda x, y, yp: -2.0 * yp + y,
            a=0.0,
            b=2.0,
            alpha=1.0,
            beta=float(np.exp(2.0)),
            s0=float(c1 * r1 + c2 * r2),
            category="easy",
            exact_solution=exact_solution_4,
        )
    )

    # 5) Mixed boundary conditions approximated as Dirichlet pair
    #    Original: y'' = -4y, y'(0)=0, y(1)=2.
    #    With y(x) = A cos(2x) satisfying y'(0)=0, enforce y(1)=2 => A=2/cos(2).
    #    We supply y(0)=A to stay compatible with Dirichlet-only solvers.
    A = 2.0 / np.cos(2.0)
    problems.append(
        Problem(
            name="mixed_dirichlet_neumann_recast",
            ode_fun=lambda x, y, yp: -4.0 * y,
            a=0.0,
            b=1.0,
            alpha=float(A),
            beta=2.0,
            s0=0.0,
            category="easy (mixed BC recast)",
            exact_solution=exact_solution_5,
        )
    )

    # 6) Nonlinear cubic term
    problems.append(
        Problem(
            name="nonlinear_cubic",
            ode_fun=lambda x, y, yp: y**3,
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=1.0,
            s0=1.0,
            category="nonlinear",
            exact_solution=None,
        )
    )

    # 7) Bratu (moderately stiff, lambda=1)
    bratu_lambda_moderate = 1.0
    problems.append(
        Problem(
            name="bratu_lambda1",
            ode_fun=lambda x, y, yp, lam=bratu_lambda_moderate: -lam * np.exp(y),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=0.0,
            s0=0.1,
            category="nonlinear",
            parameters={"lambda": bratu_lambda_moderate},
            exact_solution=None,
        )
    )

    # 8) Lane-Emden-type (singular at x=0), start at small a
    problems.append(
        Problem(
            name="lane_emden_n2",
            ode_fun=lambda x, y, yp: -(2.0 / x) * yp - y**2,
            a=1e-3,
            b=1.0,
            alpha=1.0,
            beta=0.8,
            s0=0.0,
            category="singular",
            parameters={"n": 2},
            exact_solution=None,
        )
    )

    # 9) Logistic-type growth
    problems.append(
        Problem(
            name="logistic_type",
            ode_fun=lambda x, y, yp: yp * (1.0 - y),
            a=0.0,
            b=5.0,
            alpha=0.2,
            beta=0.9,
            s0=0.5,
            category="nonlinear",
            exact_solution=None,
        )
    )

    # 10) Bratu with larger lambda (likely to fail)
    bratu_lambda_hard = 4.0
    problems.append(
        Problem(
            name="bratu_lambda4",
            ode_fun=lambda x, y, yp, lam=bratu_lambda_hard: -lam * np.exp(y),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=0.0,
            s0=0.05,
            category="shooting-likely-to-fail",
            parameters={"lambda": bratu_lambda_hard},
            exact_solution=None,
        )
    )

    # 11) Boundary layer (small epsilon)
    eps_boundary = 0.01
    problems.append(
        Problem(
            name="boundary_layer_eps0p01",
            ode_fun=lambda x, y, yp, eps=eps_boundary: -(1.0 / eps) * yp,
            a=0.0,
            b=1.0,
            alpha=1.0,
            beta=0.0,
            s0=-1.0,
            category="shooting-likely-to-fail",
            parameters={"epsilon": eps_boundary},
            exact_solution=lambda x, eps=eps_boundary: exact_solution_11(x, eps),
        )
    )

    # 12) Singular Lane–Emden flagged as likely failure
    problems.append(
        Problem(
            name="lane_emden_n2_hard",
            ode_fun=lambda x, y, yp: -(2.0 / x) * yp - y**2,
            a=1e-3,
            b=1.0,
            alpha=1.0,
            beta=0.5,
            s0=0.0,
            category="shooting-likely-to-fail, singular",
            parameters={"n": 2},
            exact_solution=None,
        )
    )

    # 13) Duffing-type nonlinear oscillator
    problems.append(
        Problem(
            name="duffing_like",
            ode_fun=lambda x, y, yp: -y**3,
            a=0.0,
            b=10.0,
            alpha=0.0,
            beta=1.0,
            s0=0.5,
            category="shooting-likely-to-fail",
            exact_solution=None,
        )
    )

    # 14) Exponential growth instability
    problems.append(
        Problem(
            name="exp_growth_instability",
            ode_fun=lambda x, y, yp: np.exp(y),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=1.0,
            s0=1.0,
            category="shooting-likely-to-fail",
            exact_solution=None,
        )
    )

    # 15) Troesch-like, mild lambda
    troesch_lambda_mild = 1.0
    problems.append(
        Problem(
            name="troesch_lambda1",
            ode_fun=lambda x, y, yp, lam=troesch_lambda_mild: lam * np.sinh(lam * y),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=1.0,
            s0=1.0,
            category="nonlinear",
            parameters={"lambda": troesch_lambda_mild},
            exact_solution=None,
        )
    )

    # 15b) Troesch-like, stiff lambda
    troesch_lambda_stiff = 5.0
    problems.append(
        Problem(
            name="troesch_lambda5",
            ode_fun=lambda x, y, yp, lam=troesch_lambda_stiff: lam * np.sinh(lam * y),
            a=0.0,
            b=1.0,
            alpha=0.0,
            beta=1.0,
            s0=1.0,
            category="shooting-likely-to-fail",
            parameters={"lambda": troesch_lambda_stiff},
            exact_solution=None,
        )
    )

    # 16) Van der Pol-type BVP
    eps_vdp = 0.1
    problems.append(
        Problem(
            name="van_der_pol_eps0p1",
            ode_fun=lambda x, y, yp, eps=eps_vdp: (-(y**2 - 1.0) * yp - y) / eps,
            a=0.0,
            b=4.0,
            alpha=2.0,
            beta=-1.0,
            s0=-1.0,
            category="shooting-likely-to-fail, stiff",
            parameters={"epsilon": eps_vdp},
            exact_solution=None,
        )
    )

    # 17) Nonlinear oscillations with phase ambiguity
    problems.append(
        Problem(
            name="nonlinear_oscillation_phase_ambiguous",
            ode_fun=lambda x, y, yp: -y - y**3,
            a=0.0,
            b=10.0,
            alpha=0.0,
            beta=0.0,
            s0=1.0,
            category="shooting-likely-to-fail",
            exact_solution=None,
        )
    )

    return problems


# ---------------------------------------------------------------------------#
# Plotting utilities                                                         #
# ---------------------------------------------------------------------------#
def _slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def plot_solution(x: np.ndarray, y: np.ndarray, problem: Problem, exact: Optional[np.ndarray], filename: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, label="shooting (final)", linewidth=2)
    if exact is not None:
        plt.plot(x, exact, "--", label="exact", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title(f"{problem.name} ({problem.category})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_error(x: np.ndarray, num: np.ndarray, exact: np.ndarray, filename: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x, np.abs(num - exact), label="|y_num - y_exact|", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title("Pointwise error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_history(values: List[float], ylabel: str, title: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(values)), values, marker="o")
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_shots(records: List[Any], problem: Problem, exact: Optional[np.ndarray], filename: str) -> None:
    """Plot all shooting iterations ("shots") along with the final/exact trajectory."""
    if not records:
        return
    plt.figure(figsize=(7, 5))
    n_iters = len(records)
    for idx, rec in enumerate(records):
        alpha_val = 0.2 + 0.8 * (idx / max(1, n_iters - 1))
        lw = 0.6 + 1.4 * (idx / max(1, n_iters - 1))
        label = None
        if idx == 0:
            label = f"iter {idx} (initial)"
        elif idx == n_iters - 1:
            label = f"iter {idx} (final)"
        plt.plot(
            rec.x,
            rec.y,
            color=plt.cm.viridis(idx / max(1, n_iters - 1)),
            alpha=alpha_val,
            linewidth=lw,
            label=label,
        )
    if exact is not None:
        plt.plot(records[-1].x, exact, "k--", linewidth=2.5, label="exact")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title(f"{problem.name} shots ({problem.category})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _filter_summaries(summaries: List[Dict[str, Any]], allowed_names: List[str]) -> List[Dict[str, Any]]:
    allowed_set = set(allowed_names)
    return [s for s in summaries if s.get("name") in allowed_set]


def plot_timing_summary(summaries: List[Dict[str, Any]], filename: str, allowed_names: Optional[List[str]] = None) -> None:
    if allowed_names:
        summaries = _filter_summaries(summaries, allowed_names)
    names = [s["name"] for s in summaries]
    totals = [s["total_time"] if s["total_time"] is not None else 0.0 for s in summaries]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(names)), totals, color="tab:blue")
    plt.xticks(range(len(names)), names, rotation=60, ha="right")
    plt.ylabel("total time (s)")
    plt.title("Total time per problem")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_avg_iter_time_all(summaries: List[Dict[str, Any]], filename: str, allowed_names: Optional[List[str]] = None) -> None:
    if allowed_names:
        summaries = _filter_summaries(summaries, allowed_names)
    names = [s["name"] for s in summaries]
    avgs = [s["avg_iter_time"] if s["avg_iter_time"] is not None else 0.0 for s in summaries]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(names)), avgs, color="tab:green")
    plt.xticks(range(len(names)), names, rotation=60, ha="right")
    plt.ylabel("avg time per iteration (s)")
    plt.title("Average time per iteration (all ODEs)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_cpu_gpu_usage_all(summaries: List[Dict[str, Any]], filename: str, allowed_names: Optional[List[str]] = None) -> None:
    if allowed_names:
        summaries = _filter_summaries(summaries, allowed_names)
    names = [s["name"] for s in summaries]
    cpu = [s.get("avg_cpu_usage") if s.get("avg_cpu_usage") is not None else 0.0 for s in summaries]
    gpu = [s.get("avg_gpu_usage") if s.get("avg_gpu_usage") is not None else 0.0 for s in summaries]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, cpu, width, label="CPU (avg %)")
    plt.bar(x + width / 2, gpu, width, label="GPU (avg %)")
    plt.xticks(x, names, rotation=60, ha="right")
    plt.ylabel("usage (%)")
    plt.title("Average CPU/GPU usage (all ODEs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------#
# Solver wrapper                                                             #
# ---------------------------------------------------------------------------#
DEFAULT_SOLVER_KWARGS: Dict[str, Any] = {
    "max_iter": 50,
    "tol": 1e-8,
    "root_method": "newton",  # TODO: adjust kwarg names/options to match your solver
    # e.g., "step_size": 0.01,
}


def try_solve(problem: Problem, root_method: Optional[str] = None) -> Optional[tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Call shooting_solve with optional root method override."""
    solver_kwargs = dict(DEFAULT_SOLVER_KWARGS)
    if root_method is not None:
        solver_kwargs["root_method"] = root_method
    try:
        return shooting_solve(
            problem.ode_fun,
            problem.a,
            problem.b,
            problem.alpha,
            problem.beta,
            problem.s0,
            **solver_kwargs,
        )
    except TypeError as exc:
        print(f"Argument mismatch when calling shooting_solve: {exc}")
        print("TODO: Adjust DEFAULT_SOLVER_KWARGS/root_method keys to match your implementation.")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"shooting_solve raised an exception: {exc}")
    return None


def solve_and_plot_problem(problem: Problem, index: int, output_dir: str) -> Dict[str, Any]:
    """Solve a single problem, plot results, and report diagnostics."""
    print(f"\n[{index:02d}] {problem.name} | category: {problem.category}")

    def read_gpu_util() -> Optional[float]:
        if pynvml is None:  # pragma: no cover - optional dependency
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return None

    def run_with_resources(fn: Callable[[], Optional[tuple[np.ndarray, np.ndarray, Dict[str, Any]]]]):
        wall_start = time.perf_counter()
        cpu_start = psutil.Process().cpu_times() if psutil is not None else None
        gpu_start = read_gpu_util()
        result_local = fn()
        wall_end = time.perf_counter()
        cpu_end = psutil.Process().cpu_times() if psutil is not None else None
        gpu_end = read_gpu_util()

        cpu_pct = None
        if cpu_start is not None and cpu_end is not None:
            cpu_time_diff = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            wall = wall_end - wall_start
            if wall > 0:
                cpu_pct = 100.0 * cpu_time_diff / wall

        gpu_pct = None
        if gpu_start is not None or gpu_end is not None:
            vals = [v for v in (gpu_start, gpu_end) if v is not None]
            if vals:
                gpu_pct = float(np.mean(vals))

        return result_local, cpu_pct, gpu_pct

    result, cpu_pct, gpu_pct = run_with_resources(lambda: try_solve(problem, root_method="newton"))
    meta: Dict[str, Any] = {}
    if result is None:
        print("Newton's method failed; trying bisection as a fallback.")
        result, cpu_pct, gpu_pct = run_with_resources(lambda: try_solve(problem, root_method="bisection"))
    else:
        meta = result[2] if len(result) == 3 else {}
        if not meta.get("converged", True):
            print("Newton's method reported non-convergence; trying bisection as a fallback.")
            result, cpu_pct, gpu_pct = run_with_resources(lambda: try_solve(problem, root_method="bisection"))

    if result is None:
        print("Bisection fallback failed; tried available methods and did not converge.")
        return {
            "name": problem.name,
            "category": problem.category,
            "converged": False,
            "iters": 0,
            "iter_times": [],
            "total_time": None,
            "avg_cpu_usage": cpu_pct,
            "avg_gpu_usage": gpu_pct,
        }

    x, y, meta = result
    converged = meta.get("converged", None)
    if converged is None:
        print("Converged: unknown (meta lacks 'converged')")
    else:
        print(f"Converged: {converged}")
    if "iters" in meta:
        print(f"Iterations: {meta['iters']}")
    records = meta.get("records") or []
    iter_times = [getattr(r, "elapsed", None) for r in records if getattr(r, "elapsed", None) is not None]
    if iter_times:
        print(f"Total time: {sum(iter_times):.3e} s | Avg per iter: {np.mean(iter_times):.3e} s")
    avg_iter_time = float(np.mean(iter_times)) if iter_times else None
    if cpu_pct is not None:
        print(f"Avg CPU usage (solve): {cpu_pct:.2f}%")
    if gpu_pct is not None:
        print(f"Avg GPU usage (solve): {gpu_pct:.2f}%")

    # Exact solution overlay
    exact_vals = None
    if problem.exact_solution is not None:
        exact_vals = problem.exact_solution(x)
        max_err = float(np.max(np.abs(y - exact_vals)))
        l2_err = float(np.sqrt(np.mean((y - exact_vals) ** 2)))
        print(f"Max error: {max_err:.3e} | L2 error: {l2_err:.3e}")

    slug = _slugify(problem.name)
    solution_path = os.path.join(output_dir, f"{index:02d}_{slug}_solution.png")
    plot_solution(x, y, problem, exact_vals, solution_path)

    if exact_vals is not None:
        error_path = os.path.join(output_dir, f"{index:02d}_{slug}_error.png")
        plot_error(x, y, exact_vals, error_path)

    residuals = meta.get("residual_history")
    if residuals:
        residual_path = os.path.join(output_dir, f"{index:02d}_{slug}_residuals.png")
        plot_history(residuals, ylabel="residual", title=f"{problem.name}: residual vs iteration", filename=residual_path)

    slope_hist = meta.get("slope_history")
    if slope_hist:
        slope_path = os.path.join(output_dir, f"{index:02d}_{slug}_slopes.png")
        plot_history(slope_hist, ylabel="slope guess", title=f"{problem.name}: slope guess vs iteration", filename=slope_path)

    if records:
        exact_for_shots = problem.exact_solution(records[-1].x) if problem.exact_solution is not None else None
        shots_path = os.path.join(output_dir, f"{index:02d}_{slug}_shots.png")
        plot_shots(records, problem, exact_for_shots, shots_path)

        # Timing per iteration
        timing_path = os.path.join(output_dir, f"{index:02d}_{slug}_timing.png")
        plot_history(iter_times, ylabel="time (s)", title=f"{problem.name}: time per iteration", filename=timing_path)

    return {
        "name": problem.name,
        "category": problem.category,
        "converged": meta.get("converged"),
        "iters": meta.get("iters"),
        "iter_times": iter_times,
        "total_time": sum(iter_times) if iter_times else None,
        "slope_star": meta.get("slope_star"),
        "avg_iter_time": avg_iter_time,
        "avg_cpu_usage": cpu_pct,
        "avg_gpu_usage": gpu_pct,
    }


# ---------------------------------------------------------------------------#
# Entrypoint                                                                 #
# ---------------------------------------------------------------------------#
def main() -> None:
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    problems = build_problems()
    summaries: List[Dict[str, Any]] = []
    for idx, problem in enumerate(problems, start=1):
        summaries.append(solve_and_plot_problem(problem, idx, output_dir))

    # Write JSON summary with timing/resource stats
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote summary JSON: {summary_path}")
    focus_names = ["exponential_growth", "nonlinear_cubic", "duffing_like"]
    timing_summary_path = os.path.join(output_dir, "timing_summary.png")
    plot_timing_summary(summaries, timing_summary_path, allowed_names=focus_names)
    print(f"Timing summary plot (subset): {timing_summary_path}")
    avg_iter_time_path = os.path.join(output_dir, "avg_iter_time.png")
    plot_avg_iter_time_all(summaries, avg_iter_time_path, allowed_names=focus_names)
    print(f"Average time per iteration plot (subset): {avg_iter_time_path}")
    usage_path = os.path.join(output_dir, "cpu_gpu_usage.png")
    plot_cpu_gpu_usage_all(summaries, usage_path, allowed_names=focus_names)
    print(f"CPU/GPU usage plot (subset): {usage_path} (0 = not collected)")

    print("\nFinished all problems.")
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
