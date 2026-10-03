"""Numerical electrostatic potential and field-line utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from skimage.measure import find_contours


FloatArray = NDArray[np.float64]
ElectrodeShape = Literal["circle", "ring", "rectangle"]


@dataclass(frozen=True)
class GridSpec:
    """Uniform rectangular paper grid, with coordinates measured in cm."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spacing: float

    def axes(self) -> tuple[FloatArray, FloatArray]:
        """Return inclusive x and y axes at the configured spacing."""
        if self.spacing <= 0:
            raise ValueError("Grid spacing must be positive.")
        x_intervals = (self.x_max - self.x_min) / self.spacing
        y_intervals = (self.y_max - self.y_min) / self.spacing
        if not np.isclose(x_intervals, round(x_intervals)) or not np.isclose(
            y_intervals, round(y_intervals)
        ):
            raise ValueError("Grid bounds must be integer multiples of spacing.")
        x = self.x_min + np.arange(round(x_intervals) + 1) * self.spacing
        y = self.y_min + np.arange(round(y_intervals) + 1) * self.spacing
        return x.astype(float), y.astype(float)


@dataclass(frozen=True)
class Electrode:
    """A fixed-voltage circular pin, grounded ring, or rectangular plate."""

    shape: ElectrodeShape
    voltage: float
    label: str
    center: tuple[float, float] | None = None
    radius: float = 0.0
    thickness: float = 0.0
    bounds: tuple[float, float, float, float] | None = None
    label_position: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.shape in ("circle", "ring") and self.center is None:
            raise ValueError(f"A {self.shape} electrode requires a center.")
        if self.shape == "rectangle" and self.bounds is None:
            raise ValueError("A rectangle electrode requires bounds.")
        if self.shape in ("circle", "ring") and self.radius <= 0:
            raise ValueError(f"A {self.shape} electrode requires a positive radius.")
        if self.shape == "ring" and self.thickness <= 0:
            raise ValueError("A ring electrode requires a positive thickness.")


@dataclass(frozen=True)
class PotentialSolution:
    """Potential values and the fixed-voltage masks used by the solver."""

    grid: GridSpec
    values: FloatArray
    fixed_mask: NDArray[np.bool_]
    electrode_masks: tuple[NDArray[np.bool_], ...]
    electrodes: tuple[Electrode, ...]


def electrode_mask(grid: GridSpec, electrode: Electrode) -> NDArray[np.bool_]:
    """Rasterize one electrode onto the paper grid."""
    x_values, y_values = grid.axes()
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    if electrode.shape in ("circle", "ring"):
        assert electrode.center is not None
        distance = np.hypot(
            x_grid - electrode.center[0], y_grid - electrode.center[1]
        )
        if electrode.shape == "circle":
            return distance <= electrode.radius
        return np.abs(distance - electrode.radius) <= electrode.thickness / 2

    assert electrode.bounds is not None
    x_min, x_max, y_min, y_max = electrode.bounds
    return (
        (x_grid >= x_min)
        & (x_grid <= x_max)
        & (y_grid >= y_min)
        & (y_grid <= y_max)
    )


def solve_potential(
    grid: GridSpec,
    electrodes: Sequence[Electrode],
    relative_tolerance: float,
    max_iterations: int,
) -> PotentialSolution:
    """Solve Laplace's equation with fixed electrodes and Neumann paper edges.

    The five-point graph Laplacian omits neighbors beyond the paper boundary;
    this implements zero normal derivative there without adding ghost cells.
    """
    if not electrodes:
        raise ValueError("At least one fixed-voltage electrode is required.")
    if relative_tolerance <= 0 or max_iterations <= 0:
        raise ValueError("Solver tolerance and iteration limit must be positive.")

    x_values, y_values = grid.axes()
    shape = (len(y_values), len(x_values))
    fixed_mask = np.zeros(shape, dtype=bool)
    fixed_values = np.zeros(shape, dtype=float)
    masks: list[NDArray[np.bool_]] = []

    for electrode in electrodes:
        mask = electrode_mask(grid, electrode)
        if not np.any(mask):
            raise ValueError(f"Electrode {electrode.label!r} misses the grid.")
        overlap = fixed_mask & mask
        if np.any(overlap) and not np.allclose(
            fixed_values[overlap], electrode.voltage
        ):
            raise ValueError("Electrodes with different voltages overlap on the grid.")
        fixed_values[mask] = electrode.voltage
        fixed_mask |= mask
        masks.append(mask)

    unknown_mask = ~fixed_mask
    unknown_flat = np.flatnonzero(unknown_mask)
    if unknown_flat.size == 0:
        raise ValueError("Electrodes cover the entire paper grid.")

    unknown_index = np.full(shape, -1, dtype=np.int64)
    unknown_index.flat[unknown_flat] = np.arange(unknown_flat.size)
    degree = np.zeros(shape, dtype=float)
    right_hand_side = np.zeros(unknown_flat.size, dtype=float)
    row_parts: list[NDArray[np.int64]] = []
    column_parts: list[NDArray[np.int64]] = []
    data_parts: list[FloatArray] = []

    adjacent_slices = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    )
    for source, target in adjacent_slices:
        source_unknown = unknown_mask[source]
        target_unknown = unknown_mask[target]
        degree[source] += 1
        source_indices = unknown_index[source]
        target_indices = unknown_index[target]

        unknown_neighbors = source_unknown & target_unknown
        row_parts.append(source_indices[unknown_neighbors])
        column_parts.append(target_indices[unknown_neighbors])
        data_parts.append(-np.ones(np.count_nonzero(unknown_neighbors)))

        fixed_neighbors = source_unknown & fixed_mask[target]
        right_hand_side[source_indices[fixed_neighbors]] += fixed_values[target][
            fixed_neighbors
        ]

    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    data = np.concatenate(data_parts)
    matrix = coo_matrix(
        (data, (rows, columns)), shape=(unknown_flat.size, unknown_flat.size)
    ).tocsr()
    matrix += diags(degree.flat[unknown_flat], format="csr")

    unknown_values, status = cg(
        matrix,
        right_hand_side,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=max_iterations,
    )
    if status != 0:
        detail = "iteration limit reached" if status > 0 else "solver breakdown"
        raise RuntimeError(f"Laplace solve failed: {detail} (status {status}).")

    values = fixed_values.copy()
    values.flat[unknown_flat] = unknown_values
    return PotentialSolution(
        grid=grid,
        values=values,
        fixed_mask=fixed_mask,
        electrode_masks=tuple(masks),
        electrodes=tuple(electrodes),
    )


def electric_field(solution: PotentialSolution) -> tuple[FloatArray, FloatArray]:
    """Compute E = -grad(V), returning x and y components in V/cm."""
    gradient_y, gradient_x = np.gradient(
        solution.values, solution.grid.spacing, solution.grid.spacing
    )
    return -gradient_x, -gradient_y


def equipotential_contours(
    solution: PotentialSolution,
    levels: Sequence[float],
    minimum_points: int,
) -> list[tuple[float, FloatArray]]:
    """Extract interior equipotential curves as arrays of (x_cm, y_cm)."""
    x_values, y_values = solution.grid.axes()
    electrode_voltages = {electrode.voltage for electrode in solution.electrodes}
    contours: list[tuple[float, FloatArray]] = []
    for level in levels:
        if any(np.isclose(level, voltage) for voltage in electrode_voltages):
            continue
        if not float(np.min(solution.values)) < level < float(
            np.max(solution.values)
        ):
            continue
        for contour in find_contours(solution.values, level):
            if len(contour) < minimum_points:
                continue
            points = np.column_stack(
                (
                    np.interp(contour[:, 1], np.arange(len(x_values)), x_values),
                    np.interp(contour[:, 0], np.arange(len(y_values)), y_values),
                )
            )
            contours.append((float(level), points))
    return contours


def trace_field_lines(
    grid: GridSpec,
    field_x: FloatArray,
    field_y: FloatArray,
    seed_points: Sequence[tuple[float, float]],
    destination_mask: NDArray[np.bool_],
    step_size: float,
    max_steps: int,
    minimum_field: float,
) -> list[FloatArray]:
    """Trace E-field curves with normalized RK4 steps from high-voltage seeds."""
    x_values, y_values = grid.axes()
    interpolate_x = RegularGridInterpolator(
        (y_values, x_values), field_x, bounds_error=False, fill_value=np.nan
    )
    interpolate_y = RegularGridInterpolator(
        (y_values, x_values), field_y, bounds_error=False, fill_value=np.nan
    )

    def direction(point: FloatArray) -> FloatArray | None:
        query = [[point[1], point[0]]]
        vector = np.array(
            [interpolate_x(query)[0], interpolate_y(query)[0]], dtype=float
        )
        magnitude = float(np.linalg.norm(vector))
        if not np.isfinite(magnitude) or magnitude < minimum_field:
            return None
        return vector / magnitude

    lines: list[FloatArray] = []
    for seed in seed_points:
        current = np.asarray(seed, dtype=float)
        points = [current.copy()]
        for _ in range(max_steps):
            first = direction(current)
            if first is None:
                break
            second = direction(current + step_size * first / 2)
            third = None if second is None else direction(current + step_size * second / 2)
            fourth = None if third is None else direction(current + step_size * third)
            if second is None or third is None or fourth is None:
                break
            next_point = current + (step_size / 6) * (
                first + 2 * second + 2 * third + fourth
            )
            if not (
                grid.x_min <= next_point[0] <= grid.x_max
                and grid.y_min <= next_point[1] <= grid.y_max
            ):
                break

            points.append(next_point.copy())
            current = next_point
            column = int(round((current[0] - grid.x_min) / grid.spacing))
            row = int(round((current[1] - grid.y_min) / grid.spacing))
            if destination_mask[row, column]:
                break

        if len(points) > 1:
            lines.append(np.asarray(points, dtype=float))
    return lines


def check_orthogonality(
    solution: PotentialSolution,
    field_lines: Sequence[FloatArray],
    dot_tolerance: float,
    minimum_field: float,
) -> bool:
    """Report whether traced-line tangents meet equipotentials near 90 degrees.

    The equipotential tangent is a 90-degree rotation of grad(V). The traced
    field-line tangent is compared with it using the normalized dot product.
    """
    gradient_y, gradient_x = np.gradient(
        solution.values, solution.grid.spacing, solution.grid.spacing
    )
    x_values, y_values = solution.grid.axes()
    interpolate_gradient_x = RegularGridInterpolator(
        (y_values, x_values), gradient_x, bounds_error=False, fill_value=np.nan
    )
    interpolate_gradient_y = RegularGridInterpolator(
        (y_values, x_values), gradient_y, bounds_error=False, fill_value=np.nan
    )
    absolute_dots: list[float] = []

    for line in field_lines:
        if len(line) < 3:
            continue
        for index in range(1, len(line) - 1):
            tangent = line[index + 1] - line[index - 1]
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length == 0:
                continue
            point = line[index]
            query = [[point[1], point[0]]]
            gx = float(interpolate_gradient_x(query)[0])
            gy = float(interpolate_gradient_y(query)[0])
            gradient_length = float(np.hypot(gx, gy))
            if not np.isfinite(gradient_length) or gradient_length < minimum_field:
                continue
            equipotential_tangent = np.array([-gy, gx]) / gradient_length
            absolute_dots.append(
                abs(float(np.dot(tangent / tangent_length, equipotential_tangent)))
            )

    maximum_dot = max(absolute_dots, default=float("inf"))
    passed = bool(absolute_dots) and maximum_dot <= dot_tolerance
    status = "PASS" if passed else "FAIL"
    print(
        f"Orthogonality check: {status} "
        f"(max |t_field · t_equipotential| = {maximum_dot:.3f}, "
        f"limit = {dot_tolerance:.3f})"
    )
    return passed