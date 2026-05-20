# FAMOS IMC3 Viewer and Converter

Tools for opening FAMOS/imc `.raw` and `.dat` files in imc3 format, converting them to CSV or HDF5, and inspecting HDF5 files.

## Features

- Convert a single `.raw`/`.dat` file or a folder of FAMOS files.
- Export each channel as separate CSV/HDF5 files or combine multiple channels into one output file.
- Preserve parsed metadata such as `Xdelta`, `Xoffset`, `Xunit`, `Ydelta`, `Yoffset`, `Yunit`, `Zdelta`, `Zoffset`, `Zunit`, scale, and offset when those values are present in the source file.
- Open one or more `.raw`/`.dat` files in the Qt-based viewer and save them as CSV or HDF5.
- Browse HDF5 groups, datasets, attributes, previews, and simple plots.

## Requirements

Python 3.10 or newer is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

The converter needs `numpy`, `pandas`, and `h5py`. The viewer additionally needs `PySide6`; plotting in the viewer needs `matplotlib`.

## Convert FAMOS Files

Edit the configuration block at the top of `IMC3_Converter.py`:

```python
INPUT_PATHS = [Path("input")]
OUTPUT_DIR = Path("converted")
SELECTED_FILES: list[str] = []
RECURSIVE = False
ZERO_X_OFFSET = False
SAVE_MODE = "combined"
```

Then run:

```bash
python IMC3_Converter.py
```

You can also call the converter from another Python script:

```python
from pathlib import Path

from IMC3_Converter import convert_famos_raw

result = convert_famos_raw(
    input_paths=Path("input"),
    output_dir=Path("converted"),
    selected_files=["channel_025*.raw", "channel_026*.dat"],
    recursive=False,
    zero_x_offset=False,
    save_mode="combined",
    export_csv=True,
    export_hdf5=True,
    combined_basename="measurement_001",
)
```

## View HDF5 and FAMOS Files

Start the viewer without a file:

```bash
python Viewer.py
```

Or open a file directly:

```bash
python Viewer.py converted/FAMOS_export.h5
python Viewer.py input/channel_025.raw
python Viewer.py input/channel_025.dat
```

Use `xoff = 0` to normalize FAMOS time axes to start at zero before saving. The viewer remembers this option, Auto-Plot, and the last folder between sessions.

When one FAMOS file is open, use **Save as...** to export it as CSV or HDF5. When multiple FAMOS files are open, **Save as...** lets you save them as one combined output file or as separate files in a folder. The status bar shows progress and the final save location.

## Data Notes

The parser expects FAMOS files that start with:

```text
|imc3,1;
```

It currently reads one channel per FAMOS file. Very large datasets are previewed and plotted with downsampling so the UI stays responsive.

## Disclaimer

This software is provided as-is, without any warranty or guarantee of functionality, correctness, fitness for a particular purpose, or compatibility with specific data files. Use it at your own risk and verify converted measurement data before relying on it.
