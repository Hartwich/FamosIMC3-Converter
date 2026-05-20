# -*- coding: utf-8 -*-
"""
FAMOS / imc .raw -> CSV and/or HDF5

This script reads FAMOS/imc RAW files in imc3 format, identified by the
file header:

    |imc3,1;

The RAW files contain marker blocks and binary measurement data blocks.
The parser currently reads one channel per RAW file.

Features:
    - read a single .raw file
    - read a folder containing .raw files
    - select specific files from a folder
    - export to CSV
    - export to HDF5
    - save each RAW file separately
    - save multiple RAW files into one combined CSV/HDF5 file

Required packages:
    pip install numpy pandas h5py
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import h5py


# =============================================================================
# Settings
# =============================================================================

# Single file:
# INPUT_PATHS = [Path(r"C:\measurement_data\channel_025.raw")]

# Or a folder:
INPUT_PATHS = [
    Path("input")
]

OUTPUT_DIR = Path("converted")


# Leave empty to use all .raw files in the folder.
# Exact file names and glob patterns are supported.
#
# Examples:
# SELECTED_FILES = []
SELECTED_FILES: list[str] = []
# SELECTED_FILES = ["channel_025.raw"]
# SELECTED_FILES = ["channel_025*.raw", "channel_026*.raw"]


# Search subfolders?
RECURSIVE = False


# Set the parsed x-axis offset to zero during conversion.
ZERO_X_OFFSET = False


# Export options
EXPORT_CSV = True
EXPORT_HDF5 = True


# Save mode:
#   "single"   -> each RAW file gets its own CSV/HDF5 file
#   "combined" -> all RAW files are stored in one CSV/HDF5 file
SAVE_MODE = "combined"  # "single" or "combined"


# Base name of the combined output file when SAVE_MODE = "combined"
COMBINED_BASENAME = "FAMOS_export"


# CSV options
CSV_SEPARATOR = ";"
CSV_DECIMAL = "."


# HDF5 options
HDF5_COMPRESSION = "gzip"  # None, "gzip", "lzf"

SUPPORTED_INPUT_SUFFIXES = {".raw", ".dat"}


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class FamosChannel:
    name: str
    values: np.ndarray
    time: np.ndarray

    x_delta: float
    x0: float
    x_unit: str
    y_unit: str

    raw_values: np.ndarray
    raw_scale: float
    raw_offset: float

    source_file: Path

    y_delta: float = 1.0
    y0: float = 0.0
    z_delta: float = 1.0
    z0: float = 0.0
    z_unit: str = ""
    hdf5_attrs: dict[str, object] = field(default_factory=dict)


# =============================================================================
# Helper functions
# =============================================================================

def safe_name(name: str) -> str:
    """
    Create safe names for files, CSV columns, and HDF5 datasets.
    """
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    name = name.strip("._")
    return name or "channel"


def unique_name(name: str, used: set[str]) -> str:
    """
    Return a safe name that is unique within the provided set.
    """
    base = safe_name(name)
    candidate = base
    i = 2

    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1

    used.add(candidate)
    return candidate


def paths_to_list(paths: Path | str | Iterable[Path | str]) -> list[Path]:
    """
    Accept a path, string, or iterable of paths/strings.
    """
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    return [Path(p) for p in paths]


def find_raw_files(
    input_paths: Path | str | Iterable[Path | str],
    selected_files: list[str] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """
    Collect RAW files from files or folders.

    input_paths:
        A single file, a single folder, or a list of files/folders.

    selected_files:
        Empty/None selects all .raw files in the folder.
        Otherwise, only matching file names or glob patterns are used.

    recursive:
        True searches subfolders.
    """
    selected_files = selected_files or []
    inputs = paths_to_list(input_paths)

    files: list[Path] = []

    for input_path in inputs:
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
                files.append(input_path)
            continue

        if not input_path.is_dir():
            print(f"WARNING: Path does not exist: {input_path}")
            continue

        if selected_files:
            for pattern in selected_files:
                if recursive:
                    files.extend(input_path.rglob(pattern))
                else:
                    files.extend(input_path.glob(pattern))
        else:
            for suffix in sorted(SUPPORTED_INPUT_SUFFIXES):
                pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
                files.extend(input_path.glob(pattern))

    # Remove duplicates and sort paths.
    files = sorted(set(p.resolve() for p in files))

    return files


# =============================================================================
# FAMOS/imc3 parser
# =============================================================================

def _marker_blocks(data: bytes) -> list[tuple[str, int, int, bytes]]:
    """
    Split the imc3 file into blocks.

    Markers in this format look like:
        |CN1, |CM1, |CC1, |RC5, ...

    Returns:
        List of (marker, start, end, block)
    """
    matches = list(re.finditer(rb"\|[A-Za-z]{2}\d", data))
    blocks: list[tuple[str, int, int, bytes]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)

        marker = data[start:start + 4].decode("latin1", errors="replace")
        block = data[start:end]

        blocks.append((marker, start, end, block))

    return blocks


def _first_block(
    blocks: list[tuple[str, int, int, bytes]],
    marker: str,
) -> bytes | None:
    """
    Return the first block with the requested marker.
    """
    for m, _, _, block in blocks:
        if m == marker:
            return block

    return None


def _extract_printable_name(block: bytes, fallback: str) -> str:
    """
    Extract the channel name from the CN1 block.
    """
    candidates = re.findall(rb"[A-Za-z0-9_\-. /()\[\]]{3,}", block[4:])

    if not candidates:
        return fallback

    best = max(candidates, key=len)
    name = best.decode("latin1", errors="replace").strip("\x00 ")

    return name or fallback


def _parse_axis_metadata(block: bytes | None, axis: str) -> tuple[float, float, str, dict[str, object]]:
    """
    Read axis metadata from a CC block.

    Observed layout:
        Offset 8:  Float64 delta
        Offset 16: Float64 origin
        then:      unit, for example "s"
    """
    attrs: dict[str, object] = {}

    if block is None or len(block) < 24:
        return 1.0, 0.0, "index", attrs

    delta = struct.unpack_from("<d", block, 8)[0]
    origin = struct.unpack_from("<d", block, 16)[0]

    candidates = re.findall(rb"[A-Za-z0-9_/%\\.\\-]+", block[24:])
    unit = candidates[-1].decode("latin1", errors="replace") if candidates else ""

    if np.isfinite(delta) and delta != 0:
        attrs[f"{axis}delta"] = float(delta)
    else:
        delta = 1.0

    if np.isfinite(origin):
        attrs[f"{axis}offset"] = float(origin)
    else:
        origin = 0.0

    if unit:
        attrs[f"{axis}unit"] = unit
    else:
        unit = "index" if axis == "X" else ""

    return float(delta), float(origin), unit, attrs


def _parse_cm1(block: bytes | None) -> tuple[float, float, str, int | None]:
    """
    Read scaling and y-axis unit metadata from the CM1 block.

    Scaling:
        y = raw * scale + offset

    Layout:
        Offset 8:  Float64 scale
        Offset 16: Float64 offset
        Offset 24: UInt16 unit length
        Offset 26: unit, for example "bar"
    """
    if block is None or len(block) < 24:
        return 1.0, 0.0, "", None

    value_type = struct.unpack_from("<I", block, 4)[0]
    scale = struct.unpack_from("<d", block, 8)[0]
    offset = struct.unpack_from("<d", block, 16)[0]

    unit = ""

    if len(block) >= 26:
        unit_len = struct.unpack_from("<H", block, 24)[0]
        unit_bytes = block[26:26 + unit_len]
        unit = unit_bytes.decode("latin1", errors="replace").strip("\x00 ")

    if not np.isfinite(scale):
        scale = 1.0

    if not np.isfinite(offset):
        offset = 0.0

    return float(scale), float(offset), unit, int(value_type)


def _parse_rt1_xoffset(block: bytes | None) -> float | None:
    """
    Read the absolute x-axis offset from the RT1 block when present.
    """
    if block is None or len(block) < 24:
        return None

    xoffset = struct.unpack_from("<d", block, 16)[0]
    if not np.isfinite(xoffset):
        return None

    return float(xoffset)


def _read_rc5_payload(blocks: list[tuple[str, int, int, bytes]]) -> bytes:
    """
    Read measurement payload bytes from RC5 blocks.

    Observed layout:
        4 Byte Marker: b'|RC5'
        4 Byte uint32
        4 Byte uint32
        4 Byte uint32: payload byte count
        Payload: binary sample data

    Some RC5 blocks can be administrative/index blocks.
    They are ignored when the payload length does not match.
    """
    chunks: list[bytes] = []

    for marker, _, _, block in blocks:
        if marker != "|RC5" or len(block) < 16:
            continue

        payload_nbytes = struct.unpack_from("<I", block, 12)[0]
        payload = block[16:]

        if payload_nbytes == 0:
            continue

        if len(payload) != payload_nbytes:
            continue

        chunks.append(bytes(payload))

    if not chunks:
        raise ValueError("No usable RC5 data blocks found.")

    return b"".join(chunks)


def _looks_like_float32(payload: bytes) -> bool:
    """
    Heuristic fallback for RAW files that do not expose a known type code.
    """
    if len(payload) == 0 or len(payload) % 4 != 0:
        return False

    values = np.frombuffer(payload, dtype="<f4")
    if values.size == 0 or not np.isfinite(values).all():
        return False

    abs_values = np.abs(values.astype(np.float64))
    max_abs = float(np.max(abs_values))
    median_abs = float(np.median(abs_values))

    return max_abs < 1e12 and median_abs > 1e-30


def _rc5_dtype_from_cm1(value_type: int | None, payload: bytes) -> np.dtype:
    """
    Resolve the RC5 payload dtype.

    The observed CM1 value type 7 stores samples as little-endian float32.
    Older files without a known type code fall back to a conservative
    auto-detection and then to int16.
    """
    if value_type == 7:
        return np.dtype("<f4")

    if _looks_like_float32(payload):
        return np.dtype("<f4")

    if len(payload) % 2 != 0:
        raise ValueError(f"RC5 payload has an odd length: {len(payload)} bytes")

    return np.dtype("<i2")


def _read_rc5_values(
    blocks: list[tuple[str, int, int, bytes]],
    value_type: int | None,
) -> np.ndarray:
    """
    Read measurement values from RC5 blocks.
    """
    payload = _read_rc5_payload(blocks)
    dtype = _rc5_dtype_from_cm1(value_type, payload)

    if len(payload) % dtype.itemsize != 0:
        raise ValueError(
            f"RC5 payload length {len(payload)} is not divisible by sample size {dtype.itemsize}."
        )

    return np.frombuffer(payload, dtype=dtype).copy()


def _read_rn1_values(
    blocks: list[tuple[str, int, int, bytes]],
    value_type: int | None,
) -> np.ndarray:
    """
    Read measurement values from RN1 blocks used in FAMOS .dat files.
    """
    chunks: list[bytes] = []

    for marker, _, _, block in blocks:
        if marker != "|RN1" or len(block) <= 4:
            continue
        chunks.append(bytes(block[4:]))

    if not chunks:
        raise ValueError("No usable RN1 data blocks found.")

    payload = b"".join(chunks)
    dtype = _rc5_dtype_from_cm1(value_type, payload)

    if len(payload) % dtype.itemsize != 0:
        raise ValueError(
            f"RN1 payload length {len(payload)} is not divisible by sample size {dtype.itemsize}."
        )

    return np.frombuffer(payload, dtype=dtype).copy()


def _read_imc3_values(
    blocks: list[tuple[str, int, int, bytes]],
    value_type: int | None,
) -> np.ndarray:
    """
    Read sample values from either RAW RC5 blocks or DAT RN1 blocks.
    """
    if any(marker == "|RC5" for marker, _, _, _ in blocks):
        try:
            return _read_rc5_values(blocks, value_type)
        except ValueError:
            if not any(marker == "|RN1" for marker, _, _, _ in blocks):
                raise

    return _read_rn1_values(blocks, value_type)


def read_famos_imc3_raw(path: Path | str, zero_x_offset: bool = False) -> FamosChannel:
    """
    Read a single FAMOS/imc3 RAW file.

    Returns:
        FamosChannel mit:
            channel.values
            channel.time
            channel.name
            channel.x_delta
            channel.y_unit
            ...
    """
    path = Path(path)
    data = path.read_bytes()

    if not data.startswith(b"|imc3,1;"):
        raise ValueError(
            f"{path.name}: File does not start with '|imc3,1;'. "
            "This script supports the imc3 RAW format from FAMOS/imc STUDIO."
        )

    blocks = _marker_blocks(data)

    name = _extract_printable_name(
        _first_block(blocks, "|CN1") or b"",
        fallback=path.stem,
    )

    x_delta, x0, x_unit, x_attrs = _parse_axis_metadata(_first_block(blocks, "|CC1"), "X")
    y_delta, y0, y_axis_unit, y_axis_attrs = _parse_axis_metadata(
        _first_block(blocks, "|CC2"),
        "Y",
    )
    z_delta, z0, z_unit, z_attrs = _parse_axis_metadata(_first_block(blocks, "|CC3"), "Z")

    cm1_block = _first_block(blocks, "|CM1")
    raw_scale, raw_offset, y_unit, value_type = _parse_cm1(cm1_block)

    hdf5_attrs: dict[str, object] = {}
    hdf5_attrs.update(x_attrs)
    hdf5_attrs.update(y_axis_attrs)
    hdf5_attrs.update(z_attrs)

    rt1_xoffset = _parse_rt1_xoffset(_first_block(blocks, "|RT1"))
    if rt1_xoffset is not None:
        x0 = rt1_xoffset
        hdf5_attrs["Xoffset"] = rt1_xoffset

    if cm1_block is not None and len(cm1_block) >= 24:
        hdf5_attrs["raw_scale"] = raw_scale
        hdf5_attrs["raw_offset"] = raw_offset
        if y_unit:
            hdf5_attrs["Yunit"] = y_unit
        elif y_axis_unit:
            hdf5_attrs["Yunit"] = y_axis_unit

    raw_values = _read_imc3_values(blocks, value_type)
    if raw_values.dtype.kind == "f" and raw_scale == 1.0 and raw_offset == 0.0:
        values = raw_values.copy()
    else:
        values = raw_values.astype(np.float64) * raw_scale + raw_offset

    if zero_x_offset:
        x0 = 0.0
        if "Xoffset" in hdf5_attrs:
            hdf5_attrs["Xoffset"] = 0.0
        if "X0" in hdf5_attrs:
            hdf5_attrs["X0"] = 0.0

    time = x0 + np.arange(values.size, dtype=np.float64) * x_delta

    return FamosChannel(
        name=name,
        values=values,
        time=time,
        x_delta=x_delta,
        x0=x0,
        x_unit=x_unit,
        y_unit=y_unit,
        raw_values=raw_values,
        raw_scale=raw_scale,
        raw_offset=raw_offset,
        source_file=path,
        y_delta=y_delta,
        y0=y0,
        z_delta=z_delta,
        z0=z0,
        z_unit=z_unit,
        hdf5_attrs=hdf5_attrs,
    )


def read_many_famos_raw(
    files: Iterable[Path | str],
    zero_x_offset: bool = False,
) -> list[FamosChannel]:
    """
    Read multiple RAW files.
    Invalid files are skipped and reported.
    """
    channels: list[FamosChannel] = []

    for file in files:
        file = Path(file)

        try:
            channel = read_famos_imc3_raw(file, zero_x_offset=zero_x_offset)
            channels.append(channel)

            print(
                f"OK: {file.name} | "
                f"Channel='{channel.name}' | "
                f"N={channel.values.size} | "
                f"dt={channel.x_delta:g} {channel.x_unit} | "
                f"Unit={channel.y_unit or '-'}"
            )

        except Exception as exc:
            print(f"ERROR: {file}: {exc}")

    return channels


def write_channel_hdf5_attrs(dset: h5py.Dataset, channel: FamosChannel) -> None:
    """
    Write only metadata attributes that were parsed from the RAW file.
    """
    for key, value in channel.hdf5_attrs.items():
        dset.attrs[key] = value

    dset.attrs["source_file"] = channel.source_file.name


def write_time_axis_hdf5_attrs(dset: h5py.Dataset, channel: FamosChannel, dataset_name: str) -> None:
    """
    Write metadata for an explicit time-axis dataset.
    """
    if "Xunit" in channel.hdf5_attrs:
        dset.attrs["Xunit"] = channel.hdf5_attrs["Xunit"]

    dset.attrs["description"] = f"time axis for {dataset_name}"


# =============================================================================
# Single-file export
# =============================================================================

def export_single_csv(
    channel: FamosChannel,
    output_dir: Path | str,
) -> Path:
    """
    Save one channel as a standalone CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{safe_name(channel.source_file.stem)}.csv"

    time_col = f"time_{channel.x_unit}" if channel.x_unit else "time"
    value_col = channel.name

    df = pd.DataFrame(
        {
            time_col: channel.time,
            value_col: channel.values,
        }
    )

    df.to_csv(
        out_path,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        index=False,
    )

    return out_path


def export_single_hdf5(
    channel: FamosChannel,
    output_dir: Path | str,
) -> Path:
    """
    Save one channel as a standalone HDF5 file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{safe_name(channel.source_file.stem)}.h5"

    dataset_name = safe_name(channel.name)

    with h5py.File(out_path, "w") as h5:
        dset = h5.create_dataset(
            dataset_name,
            data=channel.values,
            compression=HDF5_COMPRESSION,
        )

        write_channel_hdf5_attrs(dset, channel)

        h5.attrs["format"] = "converted_from_famos_imc3_raw"
        h5.attrs["source_file"] = channel.source_file.name

    return out_path


def export_channels_single_files(
    channels: list[FamosChannel],
    output_dir: Path | str,
    export_csv: bool = True,
    export_hdf5: bool = True,
) -> list[Path]:
    """
    Save each channel to its own output file.
    """
    written: list[Path] = []

    for channel in channels:
        if export_csv:
            out_csv = export_single_csv(channel, output_dir)
            written.append(out_csv)
            print(f"CSV written: {out_csv}")

        if export_hdf5:
            out_h5 = export_single_hdf5(channel, output_dir)
            written.append(out_h5)
            print(f"HDF5 written: {out_h5}")

    return written


# =============================================================================
# Combined export
# =============================================================================

def channels_have_same_time_axis(
    channels: list[FamosChannel],
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> bool:
    """
    Check whether all channels share the same time axis.
    """
    if not channels:
        return True

    ref = channels[0]

    for ch in channels[1:]:
        if ch.values.size != ref.values.size:
            return False

        if ch.x_unit != ref.x_unit:
            return False

        if not np.isclose(ch.x_delta, ref.x_delta, rtol=rtol, atol=atol):
            return False

        if not np.isclose(ch.x0, ref.x0, rtol=rtol, atol=atol):
            return False

    return True


def export_combined_csv(
    channels: list[FamosChannel],
    output_dir: Path | str,
    basename: str = "FAMOS_export",
) -> Path:
    """
    Save multiple channels into a combined CSV file.

    Case 1:
        All channels have the same time axis:
            time_s;channel_001;channel_002;...

    Case 2:
        Channels have different time axes:
            time_channel_001;channel_001;time_channel_002;channel_002;...
    """
    if not channels:
        raise ValueError("No channels were provided for combined CSV export.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{safe_name(basename)}.csv"

    used_names: set[str] = set()

    if channels_have_same_time_axis(channels):
        ref = channels[0]
        time_col = f"time_{ref.x_unit}" if ref.x_unit else "time"

        data = {
            time_col: ref.time,
        }

        for ch in channels:
            col_name = unique_name(ch.name, used_names)
            data[col_name] = ch.values

        df = pd.DataFrame(data)

    else:
        # Different time axes or lengths.
        # Store time and value as separate columns for each channel.
        series_list = []

        for ch in channels:
            base_name = unique_name(ch.name, used_names)

            s_time = pd.Series(ch.time, name=f"time_{base_name}_{ch.x_unit}")
            s_val = pd.Series(ch.values, name=base_name)

            series_list.append(s_time)
            series_list.append(s_val)

        df = pd.concat(series_list, axis=1)

    df.to_csv(
        out_path,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        index=False,
    )

    return out_path


def export_combined_hdf5(
    channels: list[FamosChannel],
    output_dir: Path | str,
    basename: str = "FAMOS_export",
) -> Path:
    """
    Save multiple channels into a combined HDF5 file.

    Structure:
        /channel_001
        /channel_002
        ...

    Each dataset contains these attributes:
        Xdelta
        X0
        Xunit
        Ydelta
        Y0
        Yunit
        Zdelta
        Z0
        Zunit
        raw_scale
        raw_offset
        source_file
    """
    if not channels:
        raise ValueError("No channels were provided for combined HDF5 export.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{safe_name(basename)}.h5"

    used_names: set[str] = set()

    with h5py.File(out_path, "w") as h5:
        h5.attrs["format"] = "converted_from_famos_imc3_raw"
        h5.attrs["n_channels"] = len(channels)

        for ch in channels:
            dataset_name = unique_name(ch.name, used_names)

            dset = h5.create_dataset(
                dataset_name,
                data=ch.values,
                compression=HDF5_COMPRESSION,
            )

            write_channel_hdf5_attrs(dset, ch)

            # Optional explicit time axis for consumers that do not use Xdelta/X0.
            time_name = f"{dataset_name}_time"
            time_dset = h5.create_dataset(
                time_name,
                data=ch.time,
                compression=HDF5_COMPRESSION,
            )
            write_time_axis_hdf5_attrs(time_dset, ch, dataset_name)

    return out_path


def export_channels_combined(
    channels: list[FamosChannel],
    output_dir: Path | str,
    basename: str = "FAMOS_export",
    export_csv: bool = True,
    export_hdf5: bool = True,
) -> list[Path]:
    """
    Save multiple channels into combined CSV and/or HDF5 files.
    """
    written: list[Path] = []

    if export_csv:
        out_csv = export_combined_csv(channels, output_dir, basename)
        written.append(out_csv)
        print(f"Combined CSV written: {out_csv}")

    if export_hdf5:
        out_h5 = export_combined_hdf5(channels, output_dir, basename)
        written.append(out_h5)
        print(f"Combined HDF5 written: {out_h5}")

    return written


# =============================================================================
# Main function for use from other scripts
# =============================================================================

def convert_famos_raw(
    input_paths: Path | str | Iterable[Path | str],
    output_dir: Path | str,
    selected_files: list[str] | None = None,
    recursive: bool = False,
    zero_x_offset: bool = False,
    save_mode: str = "single",
    export_csv: bool = True,
    export_hdf5: bool = True,
    combined_basename: str = "FAMOS_export",
) -> dict:
    """
    Main conversion function for use from other Python scripts.

    Example:
        result = convert_famos_raw(
            input_paths=Path("input"),
            output_dir=Path("converted"),
            selected_files=["channel_025*.raw"],
            recursive=False,
            zero_x_offset=False,
            save_mode="combined",
            export_csv=True,
            export_hdf5=True,
            combined_basename="measurement_001",
        )

    Returns:
        dict containing:
            files
            channels
            written
    """
    files = find_raw_files(
        input_paths=input_paths,
        selected_files=selected_files,
        recursive=recursive,
    )

    if not files:
        raise FileNotFoundError("No .raw or .dat files were found.")

    print(f"Found input files: {len(files)}")

    channels = read_many_famos_raw(files, zero_x_offset=zero_x_offset)

    if not channels:
        raise RuntimeError("No input file could be read successfully.")

    save_mode = save_mode.lower().strip()

    if save_mode == "single":
        written = export_channels_single_files(
            channels=channels,
            output_dir=output_dir,
            export_csv=export_csv,
            export_hdf5=export_hdf5,
        )

    elif save_mode == "combined":
        written = export_channels_combined(
            channels=channels,
            output_dir=output_dir,
            basename=combined_basename,
            export_csv=export_csv,
            export_hdf5=export_hdf5,
        )

    else:
        raise ValueError("save_mode must be 'single' or 'combined'.")

    return {
        "files": files,
        "channels": channels,
        "written": written,
    }


# =============================================================================
# Direct script execution
# =============================================================================

def main() -> None:
    result = convert_famos_raw(
        input_paths=INPUT_PATHS,
        output_dir=OUTPUT_DIR,
        selected_files=SELECTED_FILES,
        recursive=RECURSIVE,
        zero_x_offset=ZERO_X_OFFSET,
        save_mode=SAVE_MODE,
        export_csv=EXPORT_CSV,
        export_hdf5=EXPORT_HDF5,
        combined_basename=COMBINED_BASENAME,
    )

    print()
    print("Done.")
    print(f"Input files:    {len(result['files'])}")
    print(f"Input channels: {len(result['channels'])}")
    print(f"Written files:  {len(result['written'])}")

    for file in result["written"]:
        print(f"  {file}")


if __name__ == "__main__":
    main()
