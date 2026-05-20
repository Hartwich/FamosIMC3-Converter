from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it first: pip install h5py numpy"
    ) from exc

try:
    from IMC3_Converter import (
        FamosChannel,
        HDF5_COMPRESSION,
        read_famos_imc3_raw,
        safe_name,
        write_channel_hdf5_attrs,
        write_time_axis_hdf5_attrs,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing converter module or dependency. Install dependencies with: "
        "pip install -r requirements.txt"
    ) from exc

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except ImportError:
    Figure = None
    FigureCanvas = None
    NavigationToolbar = None

try:
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtGui import QCursor, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it first: pip install PySide6"
    ) from exc


@dataclass(frozen=True)
class NodeInfo:
    kind: str
    path: str
    extra: str = ""


@dataclass(frozen=True)
class PlotMetadata:
    x0: float = 0.0
    xdelta: float = 1.0
    xunit: str = ""
    yunit: str = ""
    has_x_metadata: bool = False


@dataclass
class ArrayDataset:
    name: str
    data: np.ndarray
    attrs: dict[str, object]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def chunks(self) -> None:
        return None

    @property
    def compression(self) -> None:
        return None

    @property
    def maxshape(self) -> tuple[int, ...]:
        return self.data.shape

    def __getitem__(self, key):
        return self.data[key]


class Viewer(QMainWindow):
    MAX_LINE_POINTS = 200_000
    MAX_LINE_SERIES = 16
    MAX_HEATMAP_PIXELS = 1_000_000
    MAX_HEATMAP_AXIS = 2_000

    def __init__(self, initial_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Viewer")
        self.resize(1250, 780)

        self.current_file_path: Path | None = None
        self.current_file_paths: list[Path] = []
        self.current_file_kind: str | None = None
        self.h5_file: h5py.File | None = None
        self.raw_channel: FamosChannel | None = None
        self.raw_channels: list[FamosChannel] = []
        self.raw_channel_nodes: dict[str, FamosChannel] = {}
        self.array_datasets: dict[str, ArrayDataset] = {}
        self.selected_plot_dataset_path: str | None = None
        self.plot_figure = None
        self.plot_canvas = None
        self.plot_message = None
        self.settings = QSettings("FAMOSIMC3Converter", "Viewer")

        self._build_ui()

        if initial_path:
            self.load_file(initial_path)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        toolbar_layout = QHBoxLayout()
        root_layout.addLayout(toolbar_layout)

        self.open_button = QPushButton("Open file...")
        self.save_button = QPushButton("Save as...")
        self.save_button.setEnabled(False)
        self.zero_x_offset_checkbox = QCheckBox("xoff = 0")
        self.zero_x_offset_checkbox.setChecked(self._setting_bool("zero_x_offset", False))
        self.expand_button = QPushButton("Expand all")
        self.collapse_button = QPushButton("Collapse all")
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        toolbar_layout.addWidget(self.open_button)
        toolbar_layout.addWidget(self.save_button)
        toolbar_layout.addWidget(self.zero_x_offset_checkbox)
        toolbar_layout.addWidget(self.expand_button)
        toolbar_layout.addWidget(self.collapse_button)
        toolbar_layout.addWidget(self.path_edit, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        left_box = QGroupBox("Structure")
        left_layout = QVBoxLayout(left_box)
        splitter.addWidget(left_box)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Name", "Type", "Shape", "Dtype"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.setMinimumWidth(420)
        self.tree.setColumnWidth(0, 280)
        self.tree.setColumnWidth(1, 110)
        self.tree.setColumnWidth(2, 120)
        self.tree.setColumnWidth(3, 120)
        left_layout.addWidget(self.tree)

        right_box = QGroupBox("Details")
        right_layout = QVBoxLayout(right_box)
        splitter.addWidget(right_box)

        self.summary_text = self._create_text_view()
        self.summary_text.setPlaceholderText("No selection yet.")
        right_layout.addWidget(self.summary_text, stretch=0)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, stretch=1)

        self.attributes_text = self._create_text_view()
        self.preview_text = self._create_text_view()
        self.plot_tab = QWidget()
        self.tabs.addTab(self.attributes_text, "Attributes")
        self.tabs.addTab(self.preview_text, "Preview")
        self.tabs.addTab(self.plot_tab, "Plot")

        self._build_plot_tab()

        splitter.setSizes([520, 730])

        self.statusBar().showMessage("Ready.")

        self.open_button.clicked.connect(self.open_file_dialog)
        self.save_button.clicked.connect(self.save_current_raw_file)
        self.zero_x_offset_checkbox.stateChanged.connect(
            lambda _state: self._on_zero_x_offset_changed()
        )
        self.expand_button.clicked.connect(self.tree.expandAll)
        self.collapse_button.clicked.connect(self.tree.collapseAll)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

    def _build_plot_tab(self) -> None:
        layout = QVBoxLayout(self.plot_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["Auto", "Line", "Heatmap"])

        self.plot_button = QPushButton("Plot")
        self.plot_button.setEnabled(False)

        self.auto_plot_checkbox = QCheckBox("Auto-Plot")
        self.auto_plot_checkbox.setChecked(self._setting_bool("auto_plot", True))

        self.plot_status_label = QLabel("Select a numeric dataset.")
        self.plot_status_label.setWordWrap(True)

        controls_layout.addWidget(QLabel("Mode:"))
        controls_layout.addWidget(self.plot_mode_combo)
        controls_layout.addWidget(self.plot_button)
        controls_layout.addWidget(self.auto_plot_checkbox)
        controls_layout.addWidget(self.plot_status_label, stretch=1)

        if Figure is None or FigureCanvas is None or NavigationToolbar is None:
            self.plot_message = self._create_text_view()
            self.plot_message.setPlainText(
                "Plotting is not available.\n\n"
                "Install matplotlib:\n"
                "pip install matplotlib"
            )
            layout.addWidget(self.plot_message, stretch=1)
            return

        self.plot_figure = Figure(figsize=(5, 4), tight_layout=True)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)

        layout.addWidget(self.plot_toolbar)
        layout.addWidget(self.plot_canvas, stretch=1)
        self._show_plot_message("Select a numeric dataset.")

        self.plot_button.clicked.connect(lambda: self.plot_selected_dataset(switch_to_tab=True))
        self.plot_mode_combo.currentTextChanged.connect(lambda _text: self._on_plot_mode_changed())
        self.auto_plot_checkbox.stateChanged.connect(lambda _state: self._on_auto_plot_changed())

    def _create_text_view(self) -> QPlainTextEdit:
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        widget.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        widget.setFont(font)
        return widget

    def _setting_bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _last_folder(self) -> str:
        value = self.settings.value("last_folder", "")
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return str(path)
        return str(Path.cwd())

    def _remember_folder(self, path: Path) -> None:
        folder = path if path.is_dir() else path.parent
        self.settings.setValue("last_folder", str(folder))

    @contextmanager
    def _wait_cursor(self):
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def open_file_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open file",
            self._last_folder(),
            (
                "Supported files (*.h5 *.hdf5 *.hdf *.he5 *.raw *.dat);;"
                "HDF5 files (*.h5 *.hdf5 *.hdf *.he5);;"
                "FAMOS files (*.raw *.dat);;"
                "All files (*.*)"
            ),
        )
        if paths:
            self._remember_folder(Path(paths[0]))
            self.load_files(paths)

    def refresh_file(self) -> None:
        if self.current_file_kind == "raw" and self.current_file_paths:
            self.load_raw_files(self.current_file_paths)
            return

        if self.current_file_path is None:
            self.statusBar().showMessage("No file loaded.")
            return

        self.load_file(str(self.current_file_path))

    def save_current_raw_file(self) -> None:
        if not self.raw_channels:
            self.statusBar().showMessage("Open a RAW file first.")
            return

        if len(self.raw_channels) > 1:
            self.save_multiple_raw_files()
            return

        channel = self.raw_channels[0]
        default_path = channel.source_file.with_suffix(".h5")
        output, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save RAW conversion",
            str(default_path),
            "HDF5 file (*.h5);;CSV file (*.csv)",
        )
        if not output:
            return

        output_path = self._save_path_with_suffix(Path(output), selected_filter)
        self._remember_folder(output_path)

        try:
            self.statusBar().showMessage("Saving...")
            QApplication.processEvents()
            with self._wait_cursor():
                if output_path.suffix.lower() == ".csv":
                    self._save_raw_as_csv(channel, output_path)
                else:
                    self._save_raw_as_hdf5(channel, output_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                f"RAW conversion could not be saved:\n{output_path}\n\n{exc}",
            )
            self.statusBar().showMessage("Save failed.")
            return

        self.statusBar().showMessage(f"Saved: {output_path}")

    def save_multiple_raw_files(self) -> None:
        save_mode = self._ask_multi_raw_save_mode()
        if save_mode is None:
            return

        if save_mode == "combined":
            self._save_multiple_raw_combined()
            return

        self._save_multiple_raw_separate()

    def _ask_multi_raw_save_mode(self) -> str | None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Save RAW files")
        box.setText("Save the RAW files as one combined file or as separate files?")

        combined_button = box.addButton("Combined file", QMessageBox.ButtonRole.AcceptRole)
        separate_button = box.addButton("Separate files", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()

        clicked_button = box.clickedButton()
        if clicked_button is combined_button:
            return "combined"
        if clicked_button is separate_button:
            return "separate"
        return None

    def _save_multiple_raw_combined(self) -> None:
        default_path = self._default_combined_output_path()
        output, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save combined RAW conversion",
            str(default_path),
            "HDF5 file (*.h5);;CSV file (*.csv)",
        )
        if not output:
            return

        output_path = self._save_path_with_suffix(Path(output), selected_filter)
        self._remember_folder(output_path)

        try:
            self.statusBar().showMessage("Saving...")
            QApplication.processEvents()
            with self._wait_cursor():
                if output_path.suffix.lower() == ".csv":
                    self._save_raw_channels_combined_csv(self.raw_channels, output_path)
                else:
                    self._save_raw_channels_combined_hdf5(self.raw_channels, output_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                f"RAW conversions could not be saved:\n{output_path}\n\n{exc}",
            )
            self.statusBar().showMessage("Save failed.")
            return

        self.statusBar().showMessage(f"Saved combined file: {output_path}")

    def _save_multiple_raw_separate(self) -> None:
        output_dir_text = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self._default_output_dir()),
        )
        if not output_dir_text:
            return

        output_format = self._ask_separate_raw_format()
        if output_format is None:
            return

        output_dir = Path(output_dir_text)
        self._remember_folder(output_dir)

        try:
            self.statusBar().showMessage("Saving...")
            QApplication.processEvents()
            with self._wait_cursor():
                written = self._save_raw_channels_separate(
                    self.raw_channels,
                    output_dir,
                    output_format,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                f"RAW conversions could not be saved:\n{output_dir}\n\n{exc}",
            )
            self.statusBar().showMessage("Save failed.")
            return

        self.statusBar().showMessage(f"Saved {len(written)} files to {output_dir}")

    def _ask_separate_raw_format(self) -> str | None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Save format")
        box.setText("Choose the output format for separate files.")

        hdf5_button = box.addButton("HDF5", QMessageBox.ButtonRole.AcceptRole)
        csv_button = box.addButton("CSV", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()

        clicked_button = box.clickedButton()
        if clicked_button is hdf5_button:
            return "hdf5"
        if clicked_button is csv_button:
            return "csv"
        return None

    def _save_path_with_suffix(self, output_path: Path, selected_filter: str) -> Path:
        if output_path.suffix:
            return output_path
        if "CSV" in selected_filter:
            return output_path.with_suffix(".csv")
        return output_path.with_suffix(".h5")

    def _save_raw_as_csv(self, channel: FamosChannel, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        time_col = f"time_{channel.x_unit}" if channel.x_unit else "time"

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([time_col, channel.name])
            writer.writerows(zip(channel.time, channel.values, strict=True))

    def _save_raw_as_hdf5(self, channel: FamosChannel, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_name = safe_name(channel.name)

        with h5py.File(output_path, "w") as h5:
            dset = h5.create_dataset(
                dataset_name,
                data=channel.values,
                compression=HDF5_COMPRESSION,
            )

            write_channel_hdf5_attrs(dset, channel)

            time_dset = h5.create_dataset(
                f"{dataset_name}_time",
                data=channel.time,
                compression=HDF5_COMPRESSION,
            )
            write_time_axis_hdf5_attrs(time_dset, channel, dataset_name)

            h5.attrs["format"] = "converted_from_famos_imc3_raw"
            h5.attrs["source_file"] = channel.source_file.name

    def _save_raw_channels_separate(
        self,
        channels: list[FamosChannel],
        output_dir: Path,
        output_format: str,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        used_names: set[str] = set()
        suffix = ".csv" if output_format == "csv" else ".h5"

        for channel in channels:
            basename = self._unique_safe_name(channel.source_file.stem, used_names)
            output_path = output_dir / f"{basename}{suffix}"

            if output_format == "csv":
                self._save_raw_as_csv(channel, output_path)
            else:
                self._save_raw_as_hdf5(channel, output_path)

            written.append(output_path)

        return written

    def _save_raw_channels_combined_csv(
        self,
        channels: list[FamosChannel],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")

            if self._channels_have_same_time_axis(channels):
                ref = channels[0]
                time_col = f"time_{ref.x_unit}" if ref.x_unit else "time"
                value_columns = [
                    self._unique_safe_name(channel.name, used_names)
                    for channel in channels
                ]
                writer.writerow([time_col, *value_columns])

                for row in zip(ref.time, *(channel.values for channel in channels), strict=True):
                    writer.writerow(row)
                return

            header: list[str] = []
            for channel in channels:
                column_name = self._unique_safe_name(channel.name, used_names)
                time_col = f"time_{column_name}_{channel.x_unit}" if channel.x_unit else f"time_{column_name}"
                header.extend([time_col, column_name])
            writer.writerow(header)

            max_length = max(channel.values.size for channel in channels)
            for index in range(max_length):
                row: list[object] = []
                for channel in channels:
                    if index < channel.values.size:
                        row.extend([channel.time[index], channel.values[index]])
                    else:
                        row.extend(["", ""])
                writer.writerow(row)

    def _save_raw_channels_combined_hdf5(
        self,
        channels: list[FamosChannel],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()

        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "converted_from_famos_imc3_raw"
            h5.attrs["n_channels"] = len(channels)
            h5.attrs["source_files"] = ";".join(channel.source_file.name for channel in channels)

            for channel in channels:
                dataset_name = self._unique_safe_name(channel.name, used_names)
                dset = h5.create_dataset(
                    dataset_name,
                    data=channel.values,
                    compression=HDF5_COMPRESSION,
                )

                write_channel_hdf5_attrs(dset, channel)

                time_dset = h5.create_dataset(
                    f"{dataset_name}_time",
                    data=channel.time,
                    compression=HDF5_COMPRESSION,
                )
                write_time_axis_hdf5_attrs(time_dset, channel, dataset_name)

    def _channels_have_same_time_axis(self, channels: list[FamosChannel]) -> bool:
        if not channels:
            return True

        ref = channels[0]
        for channel in channels[1:]:
            if channel.values.size != ref.values.size:
                return False
            if channel.x_unit != ref.x_unit:
                return False
            if not np.isclose(channel.x_delta, ref.x_delta, rtol=1e-12, atol=1e-15):
                return False
            if not np.isclose(channel.x0, ref.x0, rtol=1e-12, atol=1e-15):
                return False

        return True

    def _unique_safe_name(self, name: str, used_names: set[str]) -> str:
        base = safe_name(name)
        candidate = base
        index = 2

        while candidate in used_names:
            candidate = f"{base}_{index}"
            index += 1

        used_names.add(candidate)
        return candidate

    def _default_combined_output_path(self) -> Path:
        return self._default_output_dir() / "raw_export.h5"

    def _default_output_dir(self) -> Path:
        if self.current_file_paths:
            return self.current_file_paths[0].parent
        if self.current_file_path is not None:
            return self.current_file_path.parent
        return Path(self._last_folder())

    def load_files(self, paths: list[str]) -> None:
        resolved_paths = [Path(path).expanduser() for path in paths]

        if len(resolved_paths) == 1:
            self.load_file(str(resolved_paths[0]))
            return

        non_raw_paths = [
            path for path in resolved_paths if path.suffix.lower() not in {".raw", ".dat"}
        ]
        if non_raw_paths:
            QMessageBox.warning(
                self,
                "Unsupported selection",
                "Multiple selection currently supports RAW/DAT files only.",
            )
            self.statusBar().showMessage("Multiple selection supports RAW/DAT files only.")
            return

        self.load_raw_files(resolved_paths)

    def load_file(self, path: str) -> None:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            QMessageBox.critical(self, "File missing", f"File not found:\n{resolved}")
            self.statusBar().showMessage("File not found.")
            return

        if resolved.suffix.lower() in {".raw", ".dat"}:
            self.load_raw_file(resolved)
            return

        self.load_hdf5_file(resolved)

    def load_hdf5_file(self, path: Path) -> None:
        try:
            with self._wait_cursor():
                new_file = h5py.File(path, "r")
                self._clear_current_file()
                self.h5_file = new_file
                self.current_file_path = path
                self.current_file_paths = [path]
                self.current_file_kind = "hdf5"
                self.path_edit.setText(str(path))
                self.save_button.setEnabled(False)

                self._populate_tree()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Open failed",
                f"HDF5 file could not be opened:\n{path}\n\n{exc}",
            )
            self.statusBar().showMessage("File could not be opened.")
            return

        self.setWindowTitle(f"Viewer - {path.name}")
        self.statusBar().showMessage(f"HDF5 file loaded: {path}")
        self._remember_folder(path)

    def load_raw_file(self, path: Path) -> None:
        self.load_raw_files([path])

    def load_raw_files(self, paths: list[Path]) -> None:
        channels: list[FamosChannel] = []
        errors: list[str] = []

        with self._wait_cursor():
            for path in paths:
                try:
                    channels.append(
                        read_famos_imc3_raw(
                            path,
                            zero_x_offset=self.zero_x_offset_checkbox.isChecked(),
                        )
                    )
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")

        if not channels:
            message = "No RAW file could be opened."
            if errors:
                message = f"{message}\n\n" + "\n".join(errors[:8])
            QMessageBox.critical(self, "Open failed", message)
            self.statusBar().showMessage("No RAW file could be opened.")
            return

        with self._wait_cursor():
            self._clear_current_file()
            self.current_file_path = channels[0].source_file if len(channels) == 1 else None
            self.current_file_paths = paths
            self.current_file_kind = "raw"
            self.raw_channels = channels
            self.raw_channel = channels[0] if len(channels) == 1 else None
            self._rebuild_raw_indexes()
            self.path_edit.setText(self._raw_file_label())
            self.save_button.setEnabled(True)

            self._populate_tree()

        self.setWindowTitle(f"Viewer - {self._raw_file_label()}")
        self.statusBar().showMessage(f"RAW files loaded: {len(channels)}")
        if paths:
            self._remember_folder(paths[0])

        if errors:
            QMessageBox.warning(
                self,
                "Some RAW files were skipped",
                "Some RAW files could not be opened:\n\n" + "\n".join(errors[:8]),
            )

    def _populate_tree(self) -> None:
        if self.current_file_kind == "hdf5":
            self._populate_hdf5_tree()
            return
        if self.current_file_kind == "raw":
            self._populate_raw_tree()
            return

    def _populate_hdf5_tree(self) -> None:
        if self.h5_file is None or self.current_file_path is None:
            return

        self.tree.clear()

        root_item = QTreeWidgetItem([self.current_file_path.name, "File", "", ""])
        root_item.setData(0, Qt.UserRole, NodeInfo(kind="file", path="/"))
        self.tree.addTopLevelItem(root_item)

        root_group = self.h5_file["/"]
        root_addr = self._object_address(root_group)
        ancestors = {root_addr} if root_addr is not None else set()
        self._insert_group_children(root_item, root_group, ancestors)

        root_item.setExpanded(True)
        self.tree.setCurrentItem(root_item)

    def _populate_raw_tree(self) -> None:
        if not self.raw_channels:
            return

        self.tree.clear()

        root_type = "RAW file" if len(self.raw_channels) == 1 else "RAW collection"
        root_item = QTreeWidgetItem([self._raw_file_label(), root_type, "", ""])
        root_item.setData(0, Qt.UserRole, NodeInfo(kind="raw_file", path="/"))
        self.tree.addTopLevelItem(root_item)

        for channel_path, channel in self.raw_channel_nodes.items():
            channel_label = f"{channel.source_file.name} - {channel.name}"
            channel_item = QTreeWidgetItem(
                [
                    channel_label,
                    "Channel",
                    self._format_shape(channel.values.shape),
                    str(channel.values.dtype),
                ]
            )
            channel_item.setData(0, Qt.UserRole, NodeInfo(kind="raw_channel", path=channel_path))
            root_item.addChild(channel_item)

            for dataset_name in ("values", "time", "raw_values"):
                path = f"{channel_path}/{dataset_name}"
                dataset = self.array_datasets[path]
                item = QTreeWidgetItem(
                    [
                        dataset_name,
                        "Dataset",
                        self._format_shape(dataset.shape),
                        str(dataset.dtype),
                    ]
                )
                item.setData(0, Qt.UserRole, NodeInfo(kind="array_dataset", path=path))
                channel_item.addChild(item)

        root_item.setExpanded(True)
        first_channel_item = root_item.child(0)
        if first_channel_item is not None:
            first_channel_item.setExpanded(True)
            self.tree.setCurrentItem(first_channel_item.child(0) or first_channel_item)
        else:
            self.tree.setCurrentItem(root_item)

    def _insert_group_children(
        self,
        parent_item: QTreeWidgetItem,
        group: h5py.Group,
        ancestors: set[int],
    ) -> None:
        for name in sorted(group.keys(), key=str.lower):
            child_path = self._join_hdf5_path(group.name, name)
            link = group.get(name, getlink=True)
            obj = group.get(name, default=None)

            if obj is None:
                item = QTreeWidgetItem([name, "BrokenLink", "", ""])
                item.setData(
                    0,
                    Qt.UserRole,
                    NodeInfo(kind="broken_link", path=child_path, extra=self._describe_link(link)),
                )
                parent_item.addChild(item)
                continue

            if isinstance(obj, h5py.Group):
                addr = self._object_address(obj)
                is_cycle = addr is not None and addr in ancestors
                type_label = "Group (cycle)" if is_cycle else "Group"
                item = QTreeWidgetItem([name, type_label, "", ""])
                item.setData(0, Qt.UserRole, NodeInfo(kind="group", path=obj.name))
                parent_item.addChild(item)

                if not is_cycle:
                    next_ancestors = set(ancestors)
                    if addr is not None:
                        next_ancestors.add(addr)
                    self._insert_group_children(item, obj, next_ancestors)
                continue

            if isinstance(obj, h5py.Dataset):
                item = QTreeWidgetItem(
                    [name, "Dataset", self._format_shape(obj.shape), str(obj.dtype)]
                )
                item.setData(0, Qt.UserRole, NodeInfo(kind="dataset", path=obj.name))
                parent_item.addChild(item)
                continue

            item = QTreeWidgetItem([name, type(obj).__name__, "", ""])
            item.setData(0, Qt.UserRole, NodeInfo(kind="other", path=child_path))
            parent_item.addChild(item)

    def _on_tree_selection_changed(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return

        node: NodeInfo | None = item.data(0, Qt.UserRole)
        if node is None:
            return

        if self.current_file_kind == "raw":
            self._on_raw_tree_selection_changed(node)
            return

        if self.h5_file is None:
            return

        if node.kind == "broken_link":
            self.summary_text.setPlainText(
                f"Path: {node.path}\nType: BrokenLink\nLink: {node.extra}"
            )
            self.attributes_text.setPlainText("No attributes available.")
            self.preview_text.setPlainText("Link could not be resolved.")
            self._set_plot_target(None)
            self.statusBar().showMessage(f"Selected: {node.path}")
            return

        try:
            obj = self.h5_file[node.path]
        except Exception as exc:
            self.summary_text.setPlainText(f"Path: {node.path}\nRead error:\n{exc}")
            self.attributes_text.clear()
            self.preview_text.clear()
            self._set_plot_target(None)
            self.statusBar().showMessage("Error while reading the selection.")
            return

        self._show_object_details(obj)
        self._set_plot_target(obj if isinstance(obj, h5py.Dataset) else None)
        self.statusBar().showMessage(f"Selected: {obj.name}")

    def _on_raw_tree_selection_changed(self, node: NodeInfo) -> None:
        if not self.raw_channels:
            return

        if node.kind == "raw_file":
            self._show_raw_file_details()
            self._set_plot_target(None)
            self.statusBar().showMessage(f"Selected: {self._raw_file_label()}")
            return

        if node.kind == "raw_channel":
            channel = self.raw_channel_nodes.get(node.path)
            if channel is None:
                return
            self._show_raw_channel_details(channel)
            self._set_plot_target(self.array_datasets.get(f"{node.path}/values"))
            self.statusBar().showMessage(f"Selected channel: {channel.name}")
            return

        if node.kind == "array_dataset":
            dataset = self.array_datasets.get(node.path)
            if dataset is None:
                self.summary_text.setPlainText(f"Path: {node.path}\nRead error: dataset missing")
                self.attributes_text.clear()
                self.preview_text.clear()
                self._set_plot_target(None)
                return

            self._show_object_details(dataset)
            self._set_plot_target(dataset)
            self.statusBar().showMessage(f"Selected: {dataset.name}")

    def _show_raw_file_details(self) -> None:
        summary_lines = [
            f"File: {self._raw_file_label()}",
            "Type: FAMOS/imc3 RAW",
            f"Channels: {len(self.raw_channels)}",
            f"Samples total: {sum(channel.values.size for channel in self.raw_channels)}",
        ]
        self.summary_text.setPlainText("\n".join(summary_lines))
        self.attributes_text.setPlainText(self._format_raw_collection_metadata())
        self.preview_text.setPlainText("Open a channel or dataset to preview values.")

    def _show_raw_channel_details(self, channel: FamosChannel) -> None:
        summary_lines = [
            f"File: {channel.source_file.name}",
            f"Channel: {channel.name}",
            f"Samples: {channel.values.size}",
            f"X delta: {channel.x_delta:g}",
            f"X0: {channel.x0:g}",
            f"X unit: {channel.x_unit or '-'}",
            f"Y unit: {channel.y_unit or '-'}",
            f"Raw scale: {channel.raw_scale:g}",
            f"Raw offset: {channel.raw_offset:g}",
        ]
        self.summary_text.setPlainText("\n".join(summary_lines))
        self.attributes_text.setPlainText(self._format_channel_metadata(channel))
        self.preview_text.setPlainText(self._raw_channel_preview(channel))

    def _show_object_details(self, obj: h5py.Group | h5py.Dataset | ArrayDataset) -> None:
        summary_lines = [
            f"File: {self._current_file_label()}",
            f"Path: {obj.name}",
            f"Type: {self._object_type_name(obj)}",
            f"Attributes: {len(obj.attrs)}",
        ]

        if isinstance(obj, h5py.Group):
            summary_lines.append(f"Children: {len(obj)}")
            preview = self._group_preview(obj)
        else:
            summary_lines.extend(
                [
                    f"Shape: {self._format_shape(obj.shape)}",
                    f"Dtype: {obj.dtype}",
                    f"Elements: {obj.size}",
                    f"Raw data: {self._format_bytes(obj.nbytes)}",
                    f"Chunks: {obj.chunks if obj.chunks is not None else 'contiguous'}",
                    f"Compression: {obj.compression if obj.compression else 'none'}",
                    (
                        "Max shape: "
                        f"{self._format_shape(obj.maxshape)}"
                        if obj.maxshape is not None
                        else "Max shape: fixed"
                    ),
                ]
            )
            preview = self._dataset_preview(obj)

        self.summary_text.setPlainText("\n".join(summary_lines))
        self.attributes_text.setPlainText(self._format_attributes(obj.attrs))
        self.preview_text.setPlainText(preview)

    def _format_attributes(self, attrs: h5py.AttributeManager) -> str:
        if len(attrs) == 0:
            return "No attributes present."

        lines: list[str] = []
        for key in sorted(attrs.keys(), key=str.lower):
            try:
                lines.append(f"{key} = {self._format_value(attrs[key])}")
            except Exception as exc:
                lines.append(f"{key} = <Error: {exc}>")
        return "\n".join(lines)

    def _format_channel_metadata(self, channel: FamosChannel) -> str:
        attrs = self._channel_attrs(channel)
        return "\n".join(f"{key} = {self._format_value(value)}" for key, value in attrs.items())

    def _format_raw_collection_metadata(self) -> str:
        lines = [
            f"n_channels = {len(self.raw_channels)}",
            "source_files = "
            + "; ".join(channel.source_file.name for channel in self.raw_channels),
        ]
        return "\n".join(lines)

    def _raw_channel_preview(self, channel: FamosChannel) -> str:
        limit = min(channel.values.size, 20)
        lines = ["First samples:", "", "index;time;value;raw_value"]

        for index in range(limit):
            lines.append(
                f"{index};{channel.time[index]:g};"
                f"{channel.values[index]:g};{channel.raw_values[index]}"
            )

        if channel.values.size > limit:
            lines.append("")
            lines.append(f"... {channel.values.size - limit} additional samples hidden.")

        return "\n".join(lines)

    def _rebuild_raw_indexes(self) -> None:
        self.raw_channel_nodes = {}
        self.array_datasets = {}
        used_names: set[str] = set()

        for channel in self.raw_channels:
            node_name = self._unique_safe_name(channel.source_file.stem, used_names)
            channel_path = f"/{node_name}"
            self.raw_channel_nodes[channel_path] = channel
            self.array_datasets.update(self._raw_array_datasets(channel, channel_path))

    def _raw_file_label(self) -> str:
        if len(self.raw_channels) == 1:
            return self.raw_channels[0].source_file.name
        if self.raw_channels:
            return f"{len(self.raw_channels)} RAW files"
        return "No RAW files"

    def _current_file_label(self) -> str:
        if self.current_file_kind == "raw":
            return self._raw_file_label()
        if self.current_file_path is not None:
            return str(self.current_file_path)
        return "No file"

    def _raw_array_datasets(
        self,
        channel: FamosChannel,
        channel_path: str,
    ) -> dict[str, ArrayDataset]:
        base_attrs = self._channel_attrs(channel)
        return {
            f"{channel_path}/values": ArrayDataset(
                name=f"{channel_path}/values",
                data=channel.values,
                attrs=base_attrs.copy(),
            ),
            f"{channel_path}/time": ArrayDataset(
                name=f"{channel_path}/time",
                data=channel.time,
                attrs=self._time_axis_attrs(channel),
            ),
            f"{channel_path}/raw_values": ArrayDataset(
                name=f"{channel_path}/raw_values",
                data=channel.raw_values,
                attrs=self._raw_values_attrs(channel),
            ),
        }

    def _time_axis_attrs(self, channel: FamosChannel) -> dict[str, object]:
        attrs: dict[str, object] = {
            "description": "time axis",
            "source_file": channel.source_file.name,
        }
        if "Xunit" in channel.hdf5_attrs:
            attrs["Xunit"] = channel.hdf5_attrs["Xunit"]
        return attrs

    def _raw_values_attrs(self, channel: FamosChannel) -> dict[str, object]:
        attrs: dict[str, object] = {
            "source_file": channel.source_file.name,
        }
        for key in ("raw_scale", "raw_offset"):
            if key in channel.hdf5_attrs:
                attrs[key] = channel.hdf5_attrs[key]
        return attrs

    def _channel_attrs(self, channel: FamosChannel) -> dict[str, object]:
        attrs = dict(channel.hdf5_attrs)
        attrs["source_file"] = channel.source_file.name
        return attrs

    def _group_preview(self, group: h5py.Group) -> str:
        names = sorted(group.keys(), key=str.lower)
        if not names:
            return "Empty group."

        lines = ["Contents:"]
        limit = 200
        for name in names[:limit]:
            obj = group.get(name, default=None)
            if isinstance(obj, h5py.Group):
                lines.append(f"[Group]   {name}")
            elif isinstance(obj, h5py.Dataset):
                lines.append(
                    f"[Dataset] {name}  shape={self._format_shape(obj.shape)}  dtype={obj.dtype}"
                )
            else:
                lines.append(f"[Other]   {name}")

        if len(names) > limit:
            lines.append("")
            lines.append(f"... {len(names) - limit} additional entries hidden.")

        return "\n".join(lines)

    def _dataset_preview(self, dataset: h5py.Dataset) -> str:
        if dataset.shape == ():
            try:
                value = self._read_dataset_value(dataset)
                return f"Scalar value:\n{self._format_value(value)}"
            except Exception as exc:
                return f"Scalar value could not be read:\n{exc}"

        if dataset.size == 0:
            return "Empty dataset."

        try:
            slices = self._preview_slices(dataset.shape)
            data = self._read_dataset_slice(dataset, slices)
            lines = [
                f"Preview slice: {self._format_slice_text(slices)}",
                "",
                self._format_value(data),
            ]
            if self._is_truncated(dataset.shape, slices):
                lines.extend(
                    [
                        "",
                        "Note: The preview is truncated so large datasets stay responsive.",
                    ]
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Preview could not be read:\n{exc}"

    def _read_dataset_value(self, dataset: h5py.Dataset):
        if dataset.dtype.kind in {"S", "O", "U"}:
            try:
                return dataset.asstr()[()]
            except Exception:
                return dataset[()]
        return dataset[()]

    def _read_dataset_slice(self, dataset: h5py.Dataset, slices: tuple[slice, ...]):
        if dataset.dtype.kind in {"S", "O", "U"}:
            try:
                return np.asarray(dataset.asstr()[slices])
            except Exception:
                return np.asarray(dataset[slices])
        return np.asarray(dataset[slices])

    def _set_plot_target(self, dataset: h5py.Dataset | ArrayDataset | None) -> None:
        if dataset is None:
            self.selected_plot_dataset_path = None
            self.plot_button.setEnabled(False)
            self.plot_status_label.setText("Select a numeric dataset.")
            self._show_plot_message("Select a numeric dataset.")
            return

        self.selected_plot_dataset_path = dataset.name

        if not self._is_numeric_dataset(dataset):
            self.plot_button.setEnabled(False)
            self.plot_status_label.setText(
                f"{dataset.name}: dtype {dataset.dtype} cannot be plotted."
            )
            self._show_plot_message(
                "This dataset is not numeric and cannot be plotted directly."
            )
            return

        self.plot_button.setEnabled(Figure is not None)
        self.plot_status_label.setText(
            f"{dataset.name}: {self._format_shape(dataset.shape)}, dtype {dataset.dtype}"
        )

        if self.auto_plot_checkbox.isChecked():
            self.plot_selected_dataset(switch_to_tab=False)

    def _is_numeric_dataset(self, dataset: h5py.Dataset) -> bool:
        return dataset.dtype.kind in {"b", "i", "u", "f", "c"}

    def _on_plot_mode_changed(self) -> None:
        if self.auto_plot_checkbox.isChecked():
            self.plot_selected_dataset(switch_to_tab=False)

    def _on_auto_plot_changed(self) -> None:
        self.settings.setValue("auto_plot", self.auto_plot_checkbox.isChecked())
        if self.auto_plot_checkbox.isChecked():
            self.plot_selected_dataset(switch_to_tab=False)

    def _on_zero_x_offset_changed(self) -> None:
        self.settings.setValue("zero_x_offset", self.zero_x_offset_checkbox.isChecked())
        if self.current_file_kind == "raw" and self.current_file_paths:
            self.load_raw_files(self.current_file_paths)

    def plot_selected_dataset(self, switch_to_tab: bool = True) -> None:
        if Figure is None or self.plot_figure is None or self.plot_canvas is None:
            self._show_plot_message(
                "Plotting is not available. Install matplotlib."
            )
            return

        if self.selected_plot_dataset_path is None:
            self._show_plot_message("Select a numeric dataset.")
            return

        dataset = self._current_plot_dataset()
        if dataset is None:
            self._show_plot_message("Dataset could not be read.")
            return

        if not self._is_numeric_dataset(dataset):
            self._show_plot_message("Select a numeric dataset.")
            return

        if dataset.size == 0:
            self._show_plot_message("Empty datasets cannot be plotted.")
            return

        try:
            mode = self._resolve_plot_mode(dataset)
            if mode == "heatmap":
                self._plot_heatmap(dataset)
            else:
                self._plot_line(dataset)
        except Exception as exc:
            self._show_plot_message(f"Plot could not be created:\n{exc}")
            return

        if switch_to_tab:
            self.tabs.setCurrentWidget(self.plot_tab)

    def _current_plot_dataset(self) -> h5py.Dataset | ArrayDataset | None:
        if self.selected_plot_dataset_path is None:
            return None

        if self.current_file_kind == "raw":
            return self.array_datasets.get(self.selected_plot_dataset_path)

        if self.h5_file is None:
            return None

        try:
            dataset = self.h5_file[self.selected_plot_dataset_path]
        except Exception:
            return None

        if isinstance(dataset, h5py.Dataset):
            return dataset
        return None

    def _resolve_plot_mode(self, dataset: h5py.Dataset) -> str:
        selected = self.plot_mode_combo.currentText().lower()
        if selected == "line":
            return "line"
        if selected == "heatmap":
            return "heatmap"

        non_singleton_axes = sum(1 for dim in dataset.shape if dim > 1)
        if non_singleton_axes <= 1:
            return "line"
        if dataset.shape == () or dataset.ndim == 1:
            return "line"
        if dataset.ndim == 2 and min(dataset.shape) <= self.MAX_LINE_SERIES:
            return "line"
        return "heatmap"

    def _plot_line(self, dataset: h5py.Dataset) -> None:
        metadata = self._plot_metadata(dataset)
        x_values, series, slice_text = self._line_plot_data(dataset, metadata)

        self.plot_figure.clear()
        axes = self.plot_figure.add_subplot(111)

        for label, y_values in series:
            self._plot_one_series(axes, x_values, y_values, label)

        axes.set_title(self._short_plot_title(dataset.name))
        axes.set_xlabel(self._line_x_label(metadata))
        axes.set_ylabel(self._value_label(metadata))
        axes.grid(True, alpha=0.25)
        if len(series) > 1 or np.iscomplexobj(series[0][1]):
            axes.legend(loc="best")

        self.plot_figure.tight_layout()
        self.plot_canvas.draw_idle()
        self.plot_status_label.setText(f"Line plot: {dataset.name}  Slice {slice_text}")
        self.statusBar().showMessage(f"Plot created: {dataset.name}")

    def _line_plot_data(
        self, dataset: h5py.Dataset, metadata: PlotMetadata
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], str]:
        if dataset.shape == ():
            value = np.asarray(dataset[()])
            return (
                self._scale_x_values(np.asarray([0]), metadata),
                [("Value", value.reshape(1))],
                "[()]",
            )

        shape = dataset.shape
        sample_axis = self._largest_axis(shape)
        sample_step = max(1, ceil(shape[sample_axis] / self.MAX_LINE_POINTS))

        series_axis = self._line_series_axis(shape, sample_axis)
        selector: list[int | slice] = [0] * len(shape)
        selector[sample_axis] = slice(0, shape[sample_axis], sample_step)
        if series_axis is not None:
            selector[series_axis] = slice(0, min(shape[series_axis], self.MAX_LINE_SERIES), 1)

        data = np.asarray(dataset[tuple(selector)])
        remaining_axes = [
            axis for axis, axis_selector in enumerate(selector) if isinstance(axis_selector, slice)
        ]
        sample_position = remaining_axes.index(sample_axis)
        data = np.moveaxis(data, sample_position, 0)
        data = np.asarray(data).reshape(data.shape[0], -1)

        sample_indices = np.arange(0, shape[sample_axis], sample_step)[: data.shape[0]]
        x_values = self._scale_x_values(sample_indices, metadata)
        labels = self._line_series_labels(series_axis, data.shape[1])
        series = [(labels[index], data[:, index]) for index in range(data.shape[1])]
        return x_values, series, self._format_plot_selector(selector)

    def _plot_one_series(self, axes, x_values: np.ndarray, y_values: np.ndarray, label: str) -> None:
        if np.iscomplexobj(y_values):
            axes.plot(x_values, y_values.real, label=f"{label} real")
            if np.any(np.imag(y_values)):
                axes.plot(x_values, y_values.imag, "--", label=f"{label} imag")
            return

        axes.plot(x_values, y_values, label=label)

    def _plot_heatmap(self, dataset: h5py.Dataset) -> None:
        data, selector, axis_y, axis_x, step_y, step_x = self._heatmap_plot_data(dataset)
        display_data = np.abs(data) if np.iscomplexobj(data) else data
        metadata = self._plot_metadata(dataset)
        extent = self._heatmap_extent(
            dataset, data.shape, axis_y, axis_x, step_y, step_x, metadata
        )

        self.plot_figure.clear()
        axes = self.plot_figure.add_subplot(111)
        image = axes.imshow(
            display_data,
            aspect="auto",
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )
        colorbar = self.plot_figure.colorbar(image, ax=axes)
        colorbar.set_label(self._value_label(metadata))

        title = self._short_plot_title(dataset.name)
        if np.iscomplexobj(data):
            title = f"{title} |Magnitude|"
        axes.set_title(title)
        axes.set_xlabel(self._heatmap_x_label(axis_x, step_x, metadata))
        axes.set_ylabel(f"Axis {axis_y} index" + (f" (step {step_y})" if step_y > 1 else ""))

        self.plot_figure.tight_layout()
        self.plot_canvas.draw_idle()
        self.plot_status_label.setText(
            f"Heatmap: {dataset.name}  Slice {self._format_plot_selector(selector)}"
        )
        self.statusBar().showMessage(f"Plot created: {dataset.name}")

    def _heatmap_plot_data(
        self, dataset: h5py.Dataset
    ) -> tuple[np.ndarray, list[int | slice], int, int, int, int]:
        if dataset.shape == ():
            raise ValueError("Scalar datasets cannot be shown as a heatmap.")

        shape = dataset.shape
        if len(shape) == 1:
            step = max(1, ceil(shape[0] / self.MAX_HEATMAP_AXIS))
            selector: list[int | slice] = [slice(0, shape[0], step)]
            data = np.asarray(dataset[tuple(selector)]).reshape(1, -1)
            return data, selector, 0, 0, 1, step

        axis_y, axis_x = self._heatmap_axes(shape)
        step_y, step_x = self._heatmap_steps(shape[axis_y], shape[axis_x])

        selector = [0] * len(shape)
        selector[axis_y] = slice(0, shape[axis_y], step_y)
        selector[axis_x] = slice(0, shape[axis_x], step_x)

        data = np.asarray(dataset[tuple(selector)])
        remaining_axes = [
            axis for axis, axis_selector in enumerate(selector) if isinstance(axis_selector, slice)
        ]
        y_position = remaining_axes.index(axis_y)
        x_position = remaining_axes.index(axis_x)
        data = np.moveaxis(data, [y_position, x_position], [0, 1])
        return data, selector, axis_y, axis_x, step_y, step_x

    def _largest_axis(self, shape: tuple[int, ...]) -> int:
        return max(range(len(shape)), key=lambda axis: shape[axis])

    def _line_series_axis(self, shape: tuple[int, ...], sample_axis: int) -> int | None:
        candidates = [
            axis
            for axis, dim in enumerate(shape)
            if axis != sample_axis and 1 < dim <= self.MAX_LINE_SERIES
        ]
        if candidates:
            return candidates[0]
        return None

    def _line_series_labels(self, series_axis: int | None, count: int) -> list[str]:
        if series_axis is None:
            return ["Value"]
        return [f"Axis {series_axis} index {index}" for index in range(count)]

    def _heatmap_axes(self, shape: tuple[int, ...]) -> tuple[int, int]:
        non_singleton_axes = [axis for axis, dim in enumerate(shape) if dim > 1]
        if len(non_singleton_axes) >= 2:
            largest_axes = sorted(non_singleton_axes, key=lambda axis: shape[axis], reverse=True)[:2]
            axis_y, axis_x = sorted(largest_axes)
            return axis_y, axis_x
        if len(non_singleton_axes) == 1:
            value_axis = non_singleton_axes[0]
            other_axis = 0 if value_axis != 0 else 1
            return other_axis, value_axis
        return 0, 1

    def _heatmap_steps(self, rows: int, columns: int) -> tuple[int, int]:
        step_y = max(1, ceil(rows / self.MAX_HEATMAP_AXIS))
        step_x = max(1, ceil(columns / self.MAX_HEATMAP_AXIS))

        while ceil(rows / step_y) * ceil(columns / step_x) > self.MAX_HEATMAP_PIXELS:
            if ceil(rows / step_y) >= ceil(columns / step_x):
                step_y += 1
            else:
                step_x += 1

        return step_y, step_x

    def _plot_metadata(self, dataset: h5py.Dataset) -> PlotMetadata:
        x0_attr = self._attribute_value(dataset.attrs, "Xoffset")
        if x0_attr is None:
            x0_attr = self._attribute_value(dataset.attrs, "X0")
        xdelta_attr = self._attribute_value(dataset.attrs, "Xdelta")
        xunit = self._text_attribute(dataset.attrs, "Xunit")
        yunit = self._text_attribute(dataset.attrs, "Yunit")

        x0 = self._numeric_attribute_value(x0_attr, default=0.0)
        xdelta = self._numeric_attribute_value(xdelta_attr, default=1.0)
        has_x_metadata = x0_attr is not None or xdelta_attr is not None or bool(xunit)

        return PlotMetadata(
            x0=x0,
            xdelta=xdelta,
            xunit=xunit,
            yunit=yunit,
            has_x_metadata=has_x_metadata,
        )

    def _attribute_value(self, attrs: h5py.AttributeManager, name: str):
        if name in attrs:
            return self._scalar_attribute_value(attrs[name])

        lower_name = name.lower()
        for key in attrs.keys():
            if key.lower() == lower_name:
                return self._scalar_attribute_value(attrs[key])
        return None

    def _scalar_attribute_value(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
            else:
                return None

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        return value

    def _numeric_attribute_value(self, value, default: float) -> float:
        if value is None:
            return default

        try:
            if isinstance(value, str):
                return float(value.strip().replace(",", "."))
            return float(value)
        except (TypeError, ValueError):
            return default

    def _text_attribute(self, attrs: h5py.AttributeManager, name: str) -> str:
        value = self._attribute_value(attrs, name)
        if value is None:
            return ""

        text = str(value).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1]
        return text

    def _scale_x_values(self, indices: np.ndarray, metadata: PlotMetadata) -> np.ndarray:
        return metadata.x0 + indices.astype(float) * metadata.xdelta

    def _line_x_label(self, metadata: PlotMetadata) -> str:
        label = "X" if metadata.has_x_metadata else "Index"
        return self._label_with_unit(label, metadata.xunit)

    def _heatmap_x_label(self, axis_x: int, step_x: int, metadata: PlotMetadata) -> str:
        if metadata.has_x_metadata:
            label = self._label_with_unit("X", metadata.xunit)
        else:
            label = f"Axis {axis_x} index"
        if step_x > 1:
            return f"{label} (step {step_x})"
        return label

    def _value_label(self, metadata: PlotMetadata) -> str:
        return self._label_with_unit("Value", metadata.yunit)

    def _label_with_unit(self, label: str, unit: str) -> str:
        if unit:
            return f"{label} [{unit}]"
        return label

    def _heatmap_extent(
        self,
        dataset: h5py.Dataset,
        data_shape: tuple[int, ...],
        axis_y: int,
        axis_x: int,
        step_y: int,
        step_x: int,
        metadata: PlotMetadata,
    ) -> list[float]:
        x_indices = np.arange(0, dataset.shape[axis_x], step_x)[: data_shape[1]]
        x_values = self._scale_x_values(x_indices, metadata)
        x_delta = metadata.xdelta * step_x
        x_min, x_max = self._extent_limits(x_values, x_delta)

        y_indices = np.arange(0, dataset.shape[axis_y], step_y)[: data_shape[0]]
        y_values = y_indices.astype(float)
        y_min, y_max = self._extent_limits(y_values, float(step_y))

        return [x_min, x_max, y_min, y_max]

    def _extent_limits(self, values: np.ndarray, delta: float) -> tuple[float, float]:
        if values.size == 0:
            return -0.5, 0.5

        half_step = abs(delta) / 2.0 if delta != 0 else 0.5
        lower = float(values[0]) - half_step
        upper = float(values[-1]) + half_step
        return (lower, upper) if lower <= upper else (upper, lower)

    def _format_plot_selector(self, selector: list[int | slice]) -> str:
        parts: list[str] = []
        for item in selector:
            if isinstance(item, slice):
                start = "" if item.start is None else item.start
                stop = "" if item.stop is None else item.stop
                step = "" if item.step in (None, 1) else f":{item.step}"
                parts.append(f"{start}:{stop}{step}")
            else:
                parts.append(str(item))
        return "[" + ", ".join(parts) + "]"

    def _short_plot_title(self, path: str) -> str:
        if len(path) <= 80:
            return path
        return "..." + path[-77:]

    def _show_plot_message(self, message: str) -> None:
        if hasattr(self, "plot_status_label"):
            self.plot_status_label.setText(message.replace("\n", " "))

        if self.plot_message is not None:
            self.plot_message.setPlainText(message)
            return

        if self.plot_figure is None or self.plot_canvas is None:
            return

        self.plot_figure.clear()
        axes = self.plot_figure.add_subplot(111)
        axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        axes.set_axis_off()
        self.plot_figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _preview_slices(self, shape: tuple[int, ...]) -> tuple[slice, ...]:
        slices: list[slice] = []
        for axis, dim in enumerate(shape):
            if len(shape) == 1:
                limit = min(dim, 250)
            elif axis == 0:
                limit = min(dim, 24)
            elif axis == 1:
                limit = min(dim, 12)
            else:
                limit = min(dim, 4)
            slices.append(slice(0, limit))
        return tuple(slices)

    def _format_slice_text(self, slices: tuple[slice, ...]) -> str:
        return "[" + ", ".join(f"0:{sl.stop}" for sl in slices) + "]"

    def _is_truncated(self, shape: tuple[int, ...], slices: tuple[slice, ...]) -> bool:
        return any(dim > sl.stop for dim, sl in zip(shape, slices, strict=True))

    def _format_value(self, value) -> str:
        if isinstance(value, np.ndarray):
            return np.array2string(value, threshold=200, edgeitems=3, max_line_width=120)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return repr([self._format_value(item) for item in value])
        return repr(value)

    def _object_type_name(self, obj: h5py.Group | h5py.Dataset | ArrayDataset) -> str:
        if isinstance(obj, ArrayDataset):
            return "Dataset"
        if isinstance(obj, h5py.Dataset):
            return "Dataset"
        if obj.name == "/":
            return "File root"
        return "Group"

    def _format_shape(self, shape: tuple[int, ...] | None) -> str:
        if shape is None:
            return "-"
        if shape == ():
            return "scalar"
        return " x ".join(str(dim) for dim in shape)

    def _format_bytes(self, size: int) -> str:
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                return f"{value:.2f} {unit}"
            value /= 1024.0
        return f"{size} B"

    def _object_address(self, obj: h5py.Group | h5py.Dataset) -> int | None:
        try:
            return int(h5py.h5o.get_info(obj.id).addr)
        except Exception:
            return None

    def _join_hdf5_path(self, parent_path: str, name: str) -> str:
        if parent_path == "/":
            return f"/{name}"
        return f"{parent_path}/{name}"

    def _describe_link(self, link: object) -> str:
        if isinstance(link, h5py.SoftLink):
            return f"SoftLink -> {link.path}"
        if isinstance(link, h5py.ExternalLink):
            return f"ExternalLink -> {link.filename}:{link.path}"
        if isinstance(link, h5py.HardLink):
            return "HardLink"
        return type(link).__name__

    def _clear_current_file(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            finally:
                self.h5_file = None
        self.raw_channel = None
        self.raw_channels = []
        self.raw_channel_nodes = {}
        self.array_datasets = {}
        self.current_file_path = None
        self.current_file_paths = []
        self.current_file_kind = None
        self.selected_plot_dataset_path = None
        if hasattr(self, "save_button"):
            self.save_button.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.settings.setValue("auto_plot", self.auto_plot_checkbox.isChecked())
        self.settings.setValue("zero_x_offset", self.zero_x_offset_checkbox.isChecked())
        self.settings.sync()
        self._clear_current_file()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Viewer")

    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = Viewer(initial_path=initial_path)
    viewer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
