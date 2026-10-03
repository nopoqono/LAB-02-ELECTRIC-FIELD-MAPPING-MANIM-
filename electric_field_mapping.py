"""Manim Community Edition diagrams for the Electric Field Mapping lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from manim import *
from manim import config as manim_config
from scipy.interpolate import PchipInterpolator

from solver import (
    Electrode,
    FloatArray,
    GridSpec,
    check_orthogonality,
    electric_field,
    equipotential_contours,
    solve_potential,
    trace_field_lines,
)


@dataclass(frozen=True)
class SeedSpec:
    """Seed geometry for field lines leaving a positive electrode."""

    kind: Literal["circle", "horizontal_segment"]
    count: int
    center: tuple[float, float] | None = None
    radius: float = 0.0
    y: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0


@dataclass(frozen=True)
class SceneSpec:
    """All tuneable physical settings for one diagram."""

    title: str
    electrodes: tuple[Electrode, ...]
    equipotential_levels: tuple[float, ...]
    seed: SeedSpec
    annotation: str | None = None
    contour_paths: tuple[
        tuple[float, tuple[tuple[float, float], ...]], ...
    ] = ()


_PARALLEL_PLATE_7_5_MEASURED_POINTS = (
    (4.0, 20.0),
    (4.0, 18.0),
    (5.0, 14.0),
    (7.0, 11.5),
    (20.0, 11.5),
    (22.0, 14.0),
    (23.0, 18.0),
    (23.0, 20.0),
)
_PARALLEL_PLATE_7_5_PATH = (
    *_PARALLEL_PLATE_7_5_MEASURED_POINTS,
)
_PARALLEL_PLATE_CONTOUR_PATHS = (
    (2.5, tuple((x, 20.0 - y) for x, y in _PARALLEL_PLATE_7_5_PATH)),
    (5.0, ((0.0, 10.0), (27.0, 10.0))),
    (7.5, _PARALLEL_PLATE_7_5_PATH),
)
_DIPOLE_6V_PATH = tuple(
    (13.0 - 0.09 * (y_value - 10.0) ** 2, float(y_value))
    for y_value in np.linspace(0.0, 20.0, 101)
)
_DIPOLE_4V_PATH = tuple(
    (28.0 - x_value, y_value) for x_value, y_value in _DIPOLE_6V_PATH
)


def _dipole_circle_path(
    center: tuple[float, float], radius: float
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
        )
        for angle in np.linspace(0.0, 2.0 * np.pi, 129)
    )


_DIPOLE_CENTERED_CONTOURS = (
    (1.0, _dipole_circle_path((17.0, 10.0), 0.58)),
    (2.0, _dipole_circle_path((17.0, 10.0), 1.13)),
    (8.0, _dipole_circle_path((11.0, 10.0), 1.13)),
    (9.0, _dipole_circle_path((11.0, 10.0), 0.58)),
)


CONFIG: dict[str, object] = {
    "grid": GridSpec(x_min=0, x_max=27, y_min=0, y_max=20, spacing=0.1),
    "render": {"pixel_width": 1600, "pixel_height": 900},
    "paper": {
        "x_min_cm": 0,
        "x_max_cm": 27,
        "y_min_cm": 0,
        "y_max_cm": 20,
        "centerline_x_cm": 14,
        "centerline_y_cm": 10,
        "grid_mark_interval_cm": 1,
        "tick_interval_cm": 2,
    },
    "solver": {
        "relative_tolerance": 1e-8,
        "max_iterations": 3000,
        "minimum_contour_points": 8,
        "field_step_cm": 0.06,
        "max_field_steps": 1800,
        "minimum_field_v_per_cm": 1e-7,
        "orthogonality_dot_tolerance": 0.28,
    },
    "display": {
        "paper_width": 8.8,
        "paper_center_y": -0.25,
        "title_y": 3.68,
        "legend_y": 3.32,
        "key_x": 5.75,
        "key_center_y": -0.05,
        "key_header_font_size": 16,
        "key_font_size": 13,
        "key_line_half_length": 0.18,
        "key_row_buff": 0.13,
        "key_section_buff": 0.15,
        "key_annotation_y_offset": 1.15,
        "key_annotation_font_size": 12,
        "title_font_size": 21,
        "grid_font_size": 9,
        "axis_font_size": 9,
        "label_background_width": 4,
        "outline_samples": 241,
        "paper_border_width": 1.1,
        "grid_mark_opacity": 0.85,
        "centerline_width": 1.2,
        "centerline_opacity": 0.55,
        "equipotential_width": 2.2,
        "field_width": 1.8,
        "circle_electrode_width": 2.4,
        "ring_electrode_width": 2,
        "plate_electrode_width": 1.4,
        "electrode_fill_opacity": 0.9,
        "legend_font_size": 13,
        "legend_line_half_length": 0.25,
        "legend_line_width": 2.5,
        "legend_item_buff": 0.1,
        "legend_group_buff": 0.65,
        "field_arrow_half_length": 0.22,
        "field_arrow_width": 3.2,
        "field_arrow_tip_length": 0.2,
        "pin_visual_radius_scale": 0.45,
        "pin_fill_opacity": 0.82,
        "pin_label_font_size": 8,
        "annotation_font_size": 15,
        "annotation_y_offset": 0.22,
        "x_tick_y_offset_cm": -0.58,
        "y_tick_x_offset_cm": -0.62,
    },
    "colors": {
        "background": "#FFFFFF",
        "paper": "#FFFFFF",
        "border": "#526071",
        "grid": "#8B95A1",
        "centerline": "#7A8490",
        "text": "#17212B",
        "muted": "#526071",
        "field": "#20252B",
        "electrode": "#17212B",
    },
    "electrode_voltage_colors": {
        0.0: "#3156C5",
        10.0: "#A51F3D",
    },
    "equipotential_colors": {
        0.0: "#3156C5",
        1.0: "#197EB8",
        2.0: "#168C99",
        2.5: "#168C86",
        3.0: "#238B64",
        4.0: "#5C922B",
        5.0: "#92951F",
        6.0: "#BB861B",
        7.0: "#D56C1E",
        7.5: "#D65025",
        8.0: "#CE3B32",
        9.0: "#B92D3D",
        10.0: "#A51F3D",
    },
    "point_charge": SceneSpec(
        title="Positive Point Charge",
        electrodes=(
            Electrode(
                shape="circle",
                voltage=10,
                label="10 V",
                center=(14, 10),
                radius=0.55,
                label_position=(14.85, 11.55),
            ),
            Electrode(
                shape="ring",
                voltage=0,
                label="0 V",
                center=(14, 10),
                radius=7,
                thickness=0.2,
                label_position=(21.45, 10),
            ),
        ),
        equipotential_levels=tuple(float(value) for value in range(1, 10)),
        seed=SeedSpec(kind="circle", count=20, center=(14, 10), radius=0.72),
    ),
    "dipole": SceneSpec(
        title="Dipole",
        electrodes=(
            Electrode(
                shape="circle",
                voltage=10,
                label="10 V",
                center=(11, 10),
                radius=0.35,
                label_position=(11, 9.2),
            ),
            Electrode(
                shape="circle",
                voltage=0,
                label="0 V",
                center=(17, 10),
                radius=0.35,
                label_position=(17, 9.2),
            ),
        ),
        equipotential_levels=tuple(float(value) for value in range(0, 11)),
        seed=SeedSpec(kind="circle", count=22, center=(11, 10), radius=0.52),
        contour_paths=((4.0, _DIPOLE_4V_PATH), (6.0, _DIPOLE_6V_PATH)),
    ),
    "parallel_plates": SceneSpec(
        title="Charged Parallel Plates",
        electrodes=(
            Electrode(
                shape="rectangle",
                voltage=0,
                label="0 V",
                bounds=(8, 20, 6.9, 7.1),
                label_position=(7.1, 7),
            ),
            Electrode(
                shape="rectangle",
                voltage=10,
                label="10 V",
                bounds=(8, 20, 12.9, 13.1),
                label_position=(7.1, 13),
            ),
        ),
        equipotential_levels=(2.5, 5.0, 7.5),
        seed=SeedSpec(
            kind="horizontal_segment",
            count=17,
            y=12.75,
            x_min=8.35,
            x_max=19.65,
        ),
        annotation="E ≈ ΔV/d = 10 V / 0.06 m ≈ 170 V/m",
        contour_paths=_PARALLEL_PLATE_CONTOUR_PATHS,
    ),
    "parallel_plate_one_volt_steps": False,
}

GRID = cast(GridSpec, CONFIG["grid"])
_paper_config = cast(dict[str, float], CONFIG["paper"])
PAPER = GridSpec(
    x_min=_paper_config["x_min_cm"],
    x_max=_paper_config["x_max_cm"],
    y_min=_paper_config["y_min_cm"],
    y_max=_paper_config["y_max_cm"],
    spacing=GRID.spacing,
)
_render_config = cast(dict[str, int], CONFIG["render"])
manim_config.pixel_width = _render_config["pixel_width"]
manim_config.pixel_height = _render_config["pixel_height"]
manim_config.frame_width = (
    manim_config.frame_height
    * manim_config.pixel_width
    / manim_config.pixel_height
)


def _color(name: str) -> str:
    colors = cast(dict[str, str], CONFIG["colors"])
    return colors[name]


def _equipotential_color(level: float) -> str:
    colors = cast(dict[float, str], CONFIG["equipotential_colors"])
    try:
        return colors[float(level)]
    except KeyError as error:
        available = ", ".join(f"{value:g} V" for value in sorted(colors))
        raise ValueError(
            f"No equipotential color is configured for {level:g} V. "
            f"Add it to CONFIG['equipotential_colors']; available: {available}."
        ) from error


def _electrode_color(electrode: Electrode) -> str:
    colors = cast(dict[float, str], CONFIG["electrode_voltage_colors"])
    return colors.get(float(electrode.voltage), _color("electrode"))


def _display(name: str) -> float:
    display = cast(dict[str, float | int], CONFIG["display"])
    return float(display[name])


def _paper_setting(name: str) -> float:
    settings = cast(dict[str, float | int], CONFIG["paper"])
    return float(settings[name])


def _scene_spec(name: str) -> SceneSpec:
    return cast(SceneSpec, CONFIG[name])


def _levels_for_scene(name: str, spec: SceneSpec) -> tuple[float, ...]:
    if name == "parallel_plates" and CONFIG["parallel_plate_one_volt_steps"]:
        return tuple(float(value) for value in range(1, 10))
    return spec.equipotential_levels


def _scene_point(
    x_cm: float,
    y_cm: float,
    scale: float,
    left: float,
    bottom: float,
) -> np.ndarray:
    return np.array([left + x_cm * scale, bottom + y_cm * scale, 0.0])


def _curve_points(
    points_cm: FloatArray,
    scale: float,
    left: float,
    bottom: float,
) -> np.ndarray:
    return np.column_stack(
        (left + points_cm[:, 0] * scale, bottom + points_cm[:, 1] * scale, np.zeros(len(points_cm)))
    )


def _electrode_outline(electrode: Electrode) -> FloatArray:
    if electrode.shape in ("circle", "ring"):
        assert electrode.center is not None
        angles = np.linspace(0, 2 * np.pi, int(_display("outline_samples")))
        return np.column_stack(
            (
                electrode.center[0] + electrode.radius * np.cos(angles),
                electrode.center[1] + electrode.radius * np.sin(angles),
            )
        )

    assert electrode.bounds is not None
    x_min, x_max, y_min, y_max = electrode.bounds
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ],
        dtype=float,
    )


def _make_seeds(seed: SeedSpec) -> list[tuple[float, float]]:
    if seed.kind == "circle":
        assert seed.center is not None
        angles = np.linspace(0, 2 * np.pi, seed.count, endpoint=False)
        return [
            (
                seed.center[0] + seed.radius * np.cos(angle),
                seed.center[1] + seed.radius * np.sin(angle),
            )
            for angle in angles
        ]
    return [
        (float(x_value), seed.y)
        for x_value in np.linspace(seed.x_min, seed.x_max, seed.count)
    ]


def _smooth_curve(points: np.ndarray, color: str, width: float) -> VMobject:
    curve = VMobject()
    if len(points) > 2:
        curve.set_points_smoothly(points)
    else:
        curve.set_points_as_corners(points)
    curve.set_stroke(color=color, width=width)
    return curve


def _smooth_waypoint_curve(
    points: np.ndarray, color: str, width: float
) -> VMobject:
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))
    )
    parameters = np.union1d(
        np.linspace(
            0.0,
            distances[-1],
            max(200, int(np.ceil(distances[-1] / 0.02)) + 1),
        ),
        distances,
    )
    x_curve = PchipInterpolator(distances, points[:, 0])
    y_curve = PchipInterpolator(distances, points[:, 1])
    sampled_points = np.column_stack(
        (x_curve(parameters), y_curve(parameters), np.zeros_like(parameters))
    )
    curve = VMobject()
    curve.set_points_as_corners(sampled_points)
    curve.set_stroke(color=color, width=width)
    return curve


def _make_grid(scale: float, left: float, bottom: float) -> VGroup:
    grid_group = VGroup()
    plus_template = Text(
        "+", font_size=_display("grid_font_size"), color=_color("grid")
    )
    plus_template.set_opacity(_display("grid_mark_opacity"))
    x_values, y_values = PAPER.axes()
    mark_stride = int(round(_paper_setting("grid_mark_interval_cm") / PAPER.spacing))
    for x_index in range(0, len(x_values), mark_stride):
        x_value = x_values[x_index]
        for y_index in range(0, len(y_values), mark_stride):
            y_value = y_values[y_index]
            mark = plus_template.copy()
            mark.move_to(_scene_point(x_value, y_value, scale, left, bottom))
            grid_group.add(mark)

    centerline_color = _color("centerline")
    centerline_x = _paper_setting("centerline_x_cm")
    centerline_y = _paper_setting("centerline_y_cm")
    vertical = Line(
        _scene_point(centerline_x, PAPER.y_min, scale, left, bottom),
        _scene_point(centerline_x, PAPER.y_max, scale, left, bottom),
        color=centerline_color,
        stroke_width=_display("centerline_width"),
    ).set_opacity(_display("centerline_opacity"))
    horizontal = Line(
        _scene_point(PAPER.x_min, centerline_y, scale, left, bottom),
        _scene_point(PAPER.x_max, centerline_y, scale, left, bottom),
        color=centerline_color,
        stroke_width=_display("centerline_width"),
    ).set_opacity(_display("centerline_opacity"))
    grid_group.add(vertical, horizontal)

    tick_color = _color("muted")
    tick_interval = _paper_setting("tick_interval_cm")
    x_ticks = np.arange(PAPER.x_min, PAPER.x_max + tick_interval / 2, tick_interval)
    if not np.isclose(x_ticks[-1], PAPER.x_max):
        x_ticks = np.append(x_ticks, PAPER.x_max)
    y_ticks = np.arange(PAPER.y_min, PAPER.y_max + tick_interval / 2, tick_interval)
    for x_value in x_ticks:
        label = Text(
            f"{x_value:g}", font_size=_display("axis_font_size"), color=tick_color
        )
        label.move_to(
            _scene_point(
                x_value,
                _display("x_tick_y_offset_cm"),
                scale,
                left,
                bottom,
            )
        )
        grid_group.add(label)
    for y_value in y_ticks:
        label = Text(
            f"{y_value:g}", font_size=_display("axis_font_size"), color=tick_color
        )
        label.move_to(
            _scene_point(
                _display("y_tick_x_offset_cm"),
                y_value,
                scale,
                left,
                bottom,
            )
        )
        grid_group.add(label)
    return grid_group


def _make_electrodes(
    spec: SceneSpec,
    scale: float,
    left: float,
    bottom: float,
) -> VGroup:
    group = VGroup()
    for electrode in spec.electrodes:
        electrode_color = _electrode_color(electrode)
        if electrode.shape == "circle":
            assert electrode.center is not None
            shape = Circle(
                radius=(
                    electrode.radius
                    * scale
                    * _display("pin_visual_radius_scale")
                ),
                stroke_color=electrode_color,
                stroke_width=_display("circle_electrode_width"),
                fill_color=electrode_color,
                fill_opacity=_display("pin_fill_opacity"),
            )
            shape.move_to(
                _scene_point(*electrode.center, scale, left, bottom)
            )
            group.add(shape)
        elif electrode.shape == "ring":
            assert electrode.center is not None
            shape = Circle(
                radius=electrode.radius * scale,
                stroke_color=electrode_color,
                stroke_width=_display("ring_electrode_width"),
            )
            shape.move_to(
                _scene_point(*electrode.center, scale, left, bottom)
            )
            group.add(shape)
        else:
            assert electrode.bounds is not None
            x_min, x_max, y_min, y_max = electrode.bounds
            shape = Rectangle(
                width=(x_max - x_min) * scale,
                height=(y_max - y_min) * scale,
                stroke_color=electrode_color,
                stroke_width=_display("plate_electrode_width"),
                fill_color=electrode_color,
                fill_opacity=_display("electrode_fill_opacity"),
            )
            shape.move_to(
                _scene_point(
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    scale,
                    left,
                    bottom,
                )
            )
            group.add(shape)

        if electrode.label_position is not None:
            label = Text(
                electrode.label,
                font_size=_display("pin_label_font_size"),
                color=electrode_color,
            )
            label.set_stroke(
                color=_color("paper"),
                width=_display("label_background_width"),
                background=True,
            )
            label.move_to(
                _scene_point(*electrode.label_position, scale, left, bottom)
            )
            group.add(label)
    return group


def _make_legend(levels: tuple[float, ...]) -> VGroup:
    y_position = _display("legend_y")
    group = VGroup()
    line_half_length = _display("legend_line_half_length")
    if not levels:
        raise ValueError("The legend requires at least one equipotential level.")
    sample_indices = np.linspace(
        0, len(levels) - 1, min(3, len(levels)), dtype=int
    )
    sample_levels = sorted({levels[index] for index in sample_indices})
    equipotential_samples = VGroup(
        *(
            Line(
                LEFT * line_half_length,
                RIGHT * line_half_length,
                color=_equipotential_color(level),
                stroke_width=_display("legend_line_width"),
            )
            for level in sample_levels
        )
    )
    equipotential_samples.arrange(RIGHT, buff=_display("legend_item_buff"))
    equipotential_text = Text(
        "Equipotential (V)",
        font_size=_display("legend_font_size"),
        color=_color("text"),
    )
    equipotential_group = VGroup(equipotential_samples, equipotential_text)
    equipotential_group.arrange(RIGHT, buff=_display("legend_item_buff"))

    field_arrow = Arrow(
        LEFT * line_half_length,
        RIGHT * line_half_length,
        buff=0,
        color=_color("field"),
        stroke_width=_display("legend_line_width"),
        tip_length=_display("field_arrow_tip_length"),
    )
    field_text = Text(
        "Electric field (E)",
        font_size=_display("legend_font_size"),
        color=_color("field"),
    )
    field_group = VGroup(field_arrow, field_text)
    field_group.arrange(RIGHT, buff=_display("legend_item_buff"))

    group.add(equipotential_group, field_group)
    group.arrange(RIGHT, buff=_display("legend_group_buff"))
    group.move_to([0, y_position, 0])
    return group


def _make_voltage_key(levels: tuple[float, ...]) -> VGroup:
    if not levels:
        raise ValueError("The voltage key requires at least one equipotential level.")

    header = Text(
        "Equipotential key",
        font_size=_display("key_header_font_size"),
        color=_color("text"),
    )
    rows = VGroup()
    line_half_length = _display("key_line_half_length")
    for level in sorted(set(levels)):
        sample = Line(
            LEFT * line_half_length,
            RIGHT * line_half_length,
            color=_equipotential_color(level),
            stroke_width=_display("legend_line_width"),
        )
        label = Text(
            f"{level:g} V",
            font_size=_display("key_font_size"),
            color=_equipotential_color(level),
        )
        row = VGroup(sample, label).arrange(
            RIGHT, buff=_display("legend_item_buff")
        )
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=_display("key_row_buff"))
    key = VGroup(header, rows).arrange(
        DOWN, aligned_edge=LEFT, buff=_display("key_section_buff")
    )
    key.move_to([_display("key_x"), _display("key_center_y"), 0])
    return key


class ElectricFieldDiagram(Scene):
    """Shared rendering pipeline for a configured electric-field sketch."""

    configuration_name: str

    def construct(self) -> None:
        spec = _scene_spec(self.configuration_name)
        solver_settings = cast(dict[str, float | int], CONFIG["solver"])
        display = cast(dict[str, float | int], CONFIG["display"])
        width = float(display["paper_width"])
        scale = width / (PAPER.x_max - PAPER.x_min)
        height = (PAPER.y_max - PAPER.y_min) * scale
        center_y = float(display["paper_center_y"])
        left = -(PAPER.x_max - PAPER.x_min) * scale / 2
        bottom = center_y - height / 2
        levels = _levels_for_scene(self.configuration_name, spec)

        self.camera.background_color = _color("background")
        paper = Rectangle(width=width, height=height)
        paper.set_fill(_color("paper"), opacity=1)
        paper.set_stroke(
            _color("border"), width=_display("paper_border_width")
        )
        paper.move_to([0, center_y, 0])

        solution = solve_potential(
            GRID,
            spec.electrodes,
            relative_tolerance=float(solver_settings["relative_tolerance"]),
            max_iterations=int(solver_settings["max_iterations"]),
        )
        field_x, field_y = electric_field(solution)
        destination_electrodes = [
            mask
            for electrode, mask in zip(spec.electrodes, solution.electrode_masks)
            if electrode.voltage == min(item.voltage for item in spec.electrodes)
        ]
        destination_mask = np.logical_or.reduce(destination_electrodes)
        field_lines = trace_field_lines(
            GRID,
            field_x,
            field_y,
            _make_seeds(spec.seed),
            destination_mask,
            step_size=float(solver_settings["field_step_cm"]),
            max_steps=int(solver_settings["max_field_steps"]),
            minimum_field=float(solver_settings["minimum_field_v_per_cm"]),
        )
        check_orthogonality(
            solution,
            field_lines,
            dot_tolerance=float(solver_settings["orthogonality_dot_tolerance"]),
            minimum_field=float(solver_settings["minimum_field_v_per_cm"]),
        )

        contours = equipotential_contours(
            solution,
            levels,
            minimum_points=int(solver_settings["minimum_contour_points"]),
        )
        if spec.contour_paths:
            overridden_levels = tuple(level for level, _ in spec.contour_paths)
            contours = [
                (level, points)
                for level, points in contours
                if not any(np.isclose(level, item) for item in overridden_levels)
            ]
            contours.extend(
                (level, np.array(points, dtype=float))
                for level, points in spec.contour_paths
            )
        if self.configuration_name == "dipole":
            centered_levels = tuple(level for level, _ in _DIPOLE_CENTERED_CONTOURS)
            adjusted_contours: list[tuple[float, FloatArray]] = []
            for level, points in contours:
                if any(np.isclose(level, item) for item in centered_levels):
                    continue
                adjusted_points = points.copy()
                if np.isclose(level, 7.0):
                    left_side = adjusted_points[:, 0] < 11.0
                    leftmost_x = float(np.min(adjusted_points[left_side, 0]))
                    side_scale = (11.0 - 7.9) / (11.0 - leftmost_x)
                    adjusted_points[left_side, 0] = 11.0 + (
                        adjusted_points[left_side, 0] - 11.0
                    ) * side_scale
                elif np.isclose(level, 3.0):
                    right_side = adjusted_points[:, 0] > 17.0
                    rightmost_x = float(np.max(adjusted_points[right_side, 0]))
                    side_scale = (19.8 - 17.0) / (rightmost_x - 17.0)
                    adjusted_points[right_side, 0] = 17.0 + (
                        adjusted_points[right_side, 0] - 17.0
                    ) * side_scale
                adjusted_contours.append((level, adjusted_points))
            contours = adjusted_contours + [
                (level, np.array(points, dtype=float))
                for level, points in _DIPOLE_CENTERED_CONTOURS
            ]
        for electrode in spec.electrodes:
            if any(np.isclose(electrode.voltage, level) for level in levels):
                contours.append((electrode.voltage, _electrode_outline(electrode)))
        contours.sort(
            key=lambda item: float(
                np.sum(np.linalg.norm(np.diff(item[1], axis=0), axis=1))
            )
        )

        equipotential_group = VGroup()
        for level, points_cm in contours:
            points = _curve_points(points_cm, scale, left, bottom)
            if any(np.isclose(level, path_level) for path_level, _ in spec.contour_paths):
                equipotential_group.add(
                    _smooth_waypoint_curve(
                        points,
                        _equipotential_color(level),
                        width=_display("equipotential_width"),
                    )
                )
            else:
                equipotential_group.add(
                    _smooth_curve(
                        points,
                        _equipotential_color(level),
                        width=_display("equipotential_width"),
                    )
                )
            if self.configuration_name == "parallel_plates" and level in (2.5, 7.5):
                marker_points_cm = _PARALLEL_PLATE_7_5_MEASURED_POINTS
                if level == 2.5:
                    marker_points_cm = tuple(
                        (x, 20.0 - y)
                        for x, y in _PARALLEL_PLATE_7_5_MEASURED_POINTS
                    )
                marker_points = _curve_points(
                    np.array(marker_points_cm, dtype=float), scale, left, bottom
                )
                equipotential_group.add(
                    *(
                        Dot(
                            point,
                            radius=0.022,
                            color=_equipotential_color(level),
                        )
                        for point in marker_points
                    )
                )

        field_group = VGroup()
        arrow_group = VGroup()
        for points_cm in field_lines:
            points = _curve_points(points_cm, scale, left, bottom)
            field_group.add(
                _smooth_curve(
                    points, _color("field"), width=_display("field_width")
                )
            )
            if len(points) >= 5:
                middle = len(points) // 2
                tangent = points[middle + 1] - points[middle - 1]
                tangent_length = float(np.linalg.norm(tangent[:2]))
                if tangent_length > 0:
                    unit_tangent = tangent / tangent_length
                    half_length = _display("field_arrow_half_length")
                    arrow_group.add(
                        Arrow(
                            points[middle] - half_length * unit_tangent,
                            points[middle] + half_length * unit_tangent,
                            buff=0,
                            color=_color("field"),
                            stroke_width=_display("field_arrow_width"),
                            tip_length=_display("field_arrow_tip_length"),
                        )
                    )

        title = Text(
            f"Lab 02 – Electric Field Mapping: {spec.title}",
            font_size=_display("title_font_size"),
            color=_color("text"),
        )
        title.move_to([0, _display("title_y"), 0])

        annotation_group = VGroup()
        if spec.annotation is not None:
            annotation_text = spec.annotation
            annotation_position = [
                0,
                bottom - _display("annotation_y_offset"),
                0,
            ]
            annotation_font_size = _display("annotation_font_size")
            if self.configuration_name == "parallel_plates":
                annotation_text = "E ≈ ΔV/d\n10 V / 0.06 m ≈ 170 V/m"
                annotation_position = [
                    _display("key_x"),
                    _display("key_center_y")
                    - _display("key_annotation_y_offset"),
                    0,
                ]
                annotation_font_size = _display("key_annotation_font_size")
            annotation = Text(
                annotation_text,
                font_size=annotation_font_size,
                color=_color("text"),
            )
            annotation.move_to(annotation_position)
            annotation_group.add(annotation)

        self.add(
            paper,
            _make_grid(scale, left, bottom),
            title,
            _make_legend(levels),
            _make_voltage_key(levels),
            equipotential_group,
            field_group,
            arrow_group,
            _make_electrodes(spec, scale, left, bottom),
            annotation_group,
        )


class PointCharge(ElectricFieldDiagram):
    """Positive pin surrounded by a grounded circular electrode."""

    configuration_name = "point_charge"


class Dipole(ElectricFieldDiagram):
    """Opposite point electrodes with equipotentials and connecting field lines."""

    configuration_name = "dipole"


class ParallelPlates(ElectricFieldDiagram):
    """Finite parallel plates with naturally computed fringing fields."""

    configuration_name = "parallel_plates"