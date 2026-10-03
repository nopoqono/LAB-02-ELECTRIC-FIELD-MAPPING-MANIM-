# Electric Field Mapping

Three Manim Community Edition diagrams for the PHYS 2426 University Physics II
lab. Each scene solves the two-dimensional Laplace equation on a 0.1 cm grid,
draws voltage contours in distinct configured colors with a matching voltage
key beside the plot, and traces charcoal electric field lines in the direction of
decreasing potential.

The project's environment is isolated from the parent ManimGL repository.

## Install

Python 3.10 or newer is required. From this directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, create and activate the environment with:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Render final frames

Run one command per diagram. `-s` saves the final frame as a PNG; `-p` opens it
when rendering is complete.

```sh
manim -pqh -s electric_field_mapping.py PointCharge
manim -pqh -s electric_field_mapping.py Dipole
manim -pqh -s electric_field_mapping.py ParallelPlates
```

Use `-qh` for a quick low-quality check, or omit `-p` to render without opening
the image viewer. Manim writes PNGs under `media/images/electric_field_mapping/`.

## Configuration and checks

Edit the `CONFIG` dictionary at the top of `electric_field_mapping.py` to tune
the paper, solver, colors, electrode positions, voltage levels, and field-line
seeds. The parallel-plate scene defaults to the measured 2.5 V and 7.5 V
contours; set `parallel_plate_one_volt_steps` to `True` to show inferred 1 V
increments instead.

The displayed paper and solver domain are 27 cm wide by 20 cm high, with x
ticks from 0–27 cm and y ticks from 0–20 cm. The solver uses zero-normal-gradient
(Neumann) conditions at its edges, which are symmetric around the y=10 cm
centerline. Field-line direction follows `E = -grad(V)`, so lines run from high
to low potential and cross equipotentials at right angles. Each scene prints an orthogonality check;
`PASS` means the maximum normalized tangent dot product is within the configured
tolerance.