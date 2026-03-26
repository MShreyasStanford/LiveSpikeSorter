import sys
import os
import re
from pathlib import Path
import builtins
import pickle
import psutil
import pynvml
from pynvml import NVMLError_FunctionNotFound
import GPUtil

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QUrl, QDateTime
from PyQt5.QtGui import QDesktopServices
import qtawesome as qta
import qdarkstyle

import importlib, pkgutil
import numpy as np

# Interactive plots embedded into GUI
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure

MODULE_PREFIX = "analysis_"

def discover_analyses():
    base = Path(__file__).parent
    # ensure imports work for modules in this directory
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    found = []
    for _, modname, ispkg in pkgutil.iter_modules([str(base)]):
        if not ispkg and modname.startswith(MODULE_PREFIX):
            module = importlib.import_module(modname)
            meta = getattr(module, "__analysis_metadata__", None)
            # require both metadata and a run() func
            if meta and hasattr(module, "run"):
                found.append({
                    "name": meta["name"],
                    "func": module.run,
                    "args": meta["parameters"],
                })
            else:
                print(f"[GUI] Error: Analysis module {modname} detected but is missing either metadata information or fails to implement a run(...) method.", file=sys.stderr)
    return found

analyses = discover_analyses()

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
    def write(self, text):
        if "\r" in text:
            parts = text.split("\r")
            self.textWritten.emit("\r" + parts[-1])
        else:
            self.textWritten.emit(text)
    def flush(self): pass

class PipelineWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    outputFileWritten = QtCore.pyqtSignal(str)
    def __init__(self, base_dir, pipeline_descriptions, analysis_mapping):
        super().__init__()
        self.base_dir = base_dir
        self.pipeline_descriptions = pipeline_descriptions
        self.analysis_mapping = analysis_mapping
        self._stop = False
    @QtCore.pyqtSlot()
    def run(self):
        original_open = builtins.open
        def patched_open(*args, **kwargs):
            filename = args[0] if args else kwargs.get("file")
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if any(m in mode for m in ("w","a","x")):
                self.outputFileWritten.emit(str(filename))
            return original_open(*args, **kwargs)
        builtins.open = patched_open
        try:
            for desc in self.pipeline_descriptions:
                if self._stop:
                    print("[GUI] Pipeline stopped by user.\n")
                    break
                runner = self.analysis_mapping.get(desc)
                if runner:
                    print(f"[GUI] Running {desc}...\n")
                    try:
                        runner(Path(self.base_dir))
                    except Exception:
                        import traceback
                        print(f"[GUI] Error in {desc}:\n{traceback.format_exc()}\n", file=sys.stderr)
                else:
                    print(f"[GUI] No function found for {desc}\n")
            print("[GUI] Pipeline finished.\n")
            self.finished.emit()
        finally:
            builtins.open = original_open
    def stop(self): self._stop = True

class PipelineListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
    def dragEnterEvent(self, event):
        if event.source() == self or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    def dragMoveEvent(self, event):
        if event.source() == self or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    def dropEvent(self, event):
        if event.source() is not self:
            text = event.mimeData().text()
            drop_index = self.indexAt(event.pos()).row()
            if drop_index < 0:
                self.addItem(text)
            else:
                self.insertItem(drop_index, text)
            event.acceptProposedAction()
        else:
            event.setDropAction(QtCore.Qt.MoveAction)
            super().dropEvent(event)

class AvailableListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
    def mimeData(self, items):
        mime = super().mimeData(items)
        mime.setText("\n".join(item.text() for item in items))
        return mime
    def dropEvent(self, event):
        if event.source() and isinstance(event.source(), PipelineListWidget):
            text = event.mimeData().text().strip()
            for item in event.source().findItems(text, QtCore.Qt.MatchExactly):
                event.source().takeItem(event.source().row(item))
            event.acceptProposedAction()
        else:
            event.ignore()

class MainWindow(QtWidgets.QWidget):
    STATE_FILE = Path(__file__).parent / "pipeline_state.pkl"

    def __init__(self):
        super().__init__()

        # ── Track only this process and its children ─────────────────
        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(None)
        for child in self._proc.children(recursive=True):
            child.cpu_percent(None)

        # ── Init NVML for per-process GPU memory tracking ────────────
        pynvml.nvmlInit()
        self._gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        self._gpu_total_mem = {
            i: pynvml.nvmlDeviceGetMemoryInfo(h).total
            for i, h in enumerate(self._gpu_handles)
        }

        # ── Window information ────────────
        self.setWindowTitle("Analysis Pipeline Builder")
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.setupUI()

        # ── Keyboard shortcuts ─────────────────────────────────────────
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, activated=self.run_pipeline)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self, activated=self.stop_pipeline)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+L'), self, activated=lambda: self.log.clear())

        # ── Redirect streams for logger  ─────────────────────────────────────────
        self.stdout_stream = EmittingStream(); self.stdout_stream.textWritten.connect(self.append_log)
        self._stdout = sys.stdout; sys.stdout = self.stdout_stream
        self.stderr_stream = EmittingStream(); self.stderr_stream.textWritten.connect(self.append_log)
        self._stderr = sys.stderr; sys.stderr = self.stderr_stream

        # ── In-memory persistent settings ─────────────────────────────────────────
        self.settings_cache = {}

        # ── Pickled persistent settings ─────────────────────────────────────────
        self.load_state()

        # start live resource monitor updates
        self.res_timer = QtCore.QTimer(self)
        self.res_timer.timeout.connect(self.update_resources)
        self.res_timer.start(1000)

    def setupUI(self):
        heading_font = QFont("Segoe UI", 14, QFont.Bold)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget(); left_layout = QtWidgets.QVBoxLayout(left)
        self.setStyleSheet("""
            /* Diagnostics tab styles */
            QToolButton#diagBtn {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                              stop:0 #4caf50, stop:1 #81c784);
                border-radius:4px; padding:6px 12px;
            }
            QToolButton#diagBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                              stop:0 #66bb6a, stop:1 #a5d6a7);
            }
            QProgressBar#diagProgress {
                background-color:#3a3a3a; border:1px solid #555;
                border-radius:4px; text-align:center; color:#e0e0e0;
            }
            QProgressBar#diagProgress::chunk {
                border-radius:4px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                             stop:0 #4caf50, stop:1 #81c784);
            }
            """)

        # ── Section Heading: Pipeline ──────────────────────
        pipeline_lbl = QtWidgets.QLabel("Pipeline")
        pipeline_lbl.setFont(heading_font)
        left_layout.addWidget(pipeline_lbl)

        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Output Directory:"))
        self.base_dir_edit = QtWidgets.QLineEdit(); row.addWidget(self.base_dir_edit)
        btn = QtWidgets.QPushButton("Browse"); btn.clicked.connect(self.browseBaseDir); row.addWidget(btn)
        left_layout.addLayout(row)
        frame = QtWidgets.QFrame(); frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        plo = QtWidgets.QHBoxLayout(frame)
        self.pipelineList = PipelineListWidget(); self.pipelineList.setMinimumWidth(200); self.pipelineList.setFont(QFont("Segoe UI", 12))
        self.availableList = AvailableListWidget(); self.availableList.setMinimumWidth(200); self.availableList.setFont(QFont("Segoe UI", 12))
        plo.addWidget(self.pipelineList); plo.addWidget(self.availableList)
        left_layout.addWidget(frame)
        header = QtWidgets.QLabel("Pipeline Parameters"); 
        header.setFont(heading_font)
        left_layout.addWidget(header)
        self.settingsTabs = QtWidgets.QTabWidget(); left_layout.addWidget(self.settingsTabs)

        # ── Stylized pill‐shaped button group ──────────────────────────
        toolbar_group = QtWidgets.QFrame(objectName='toolbarGroup')
        tb_layout = QtWidgets.QHBoxLayout(toolbar_group)
        tb_layout.setContentsMargins(4,4,4,4)
        self.run_btn = QtWidgets.QToolButton(objectName='runBtn')
        self.run_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.run_btn.setText('Run Pipeline')
        self.run_btn.setToolTip('Run Pipeline (Ctrl+R)')
        self.run_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.run_btn.setIconSize(QtCore.QSize(48, 48))
        self.run_btn.clicked.connect(self.run_pipeline)
        tb_layout.addWidget(self.run_btn)

        # Stop button (larger icon + text)
        self.stop_btn = QtWidgets.QToolButton(objectName='stopBtn')
        self.stop_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stop_btn.setText('Stop Pipeline')
        self.stop_btn.setToolTip('Stop Pipeline (Ctrl+S)')
        self.stop_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.stop_btn.setIconSize(QtCore.QSize(48, 48))
        self.stop_btn.clicked.connect(self.stop_pipeline)
        tb_layout.addWidget(self.stop_btn)

        # Clear Log button (larger icon + text)
        self.clear_btn = QtWidgets.QToolButton(objectName='clearBtn')
        self.clear_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        self.clear_btn.setText('Clear Log')
        self.clear_btn.setToolTip('Clear Log (Ctrl+L)')
        self.clear_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.clear_btn.setIconSize(QtCore.QSize(48, 48))
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        tb_layout.addWidget(self.clear_btn)

        # Save/Load state buttons
        self.save_btn = QtWidgets.QToolButton(objectName='saveBtn')
        self.save_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_btn.setText('Save State')
        self.save_btn.setToolTip('Save State As...')
        self.save_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.save_btn.setIconSize(QtCore.QSize(48, 48))
        self.save_btn.clicked.connect(self.save_state_as)
        tb_layout.addWidget(self.save_btn)
        self.load_btn = QtWidgets.QToolButton(objectName='loadBtn')
        self.load_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.load_btn.setText('Load State')
        self.load_btn.setToolTip('Load State...')
        self.load_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.load_btn.setIconSize(QtCore.QSize(48, 48))
        self.load_btn.clicked.connect(self.load_state_as)
        tb_layout.addWidget(self.load_btn)

        # ── Validate Layout Button ───────────────────────────────────────
        self.validate_btn = QtWidgets.QToolButton(objectName='validateBtn')
        self.validate_btn.setText('Validate file layout')
        self.validate_btn.setToolTip(
            'Check that each FilePath input matches its regex pattern.\n'
            'May take a while for large files.'
        )
        self.validate_btn.clicked.connect(self.validate_file_layout)
        tb_layout.addWidget(self.validate_btn)

        left_layout.addWidget(toolbar_group)
        # ── Live Resource Monitor ─────────────────────────────────────
        # ── Section Heading: Resources ────────────────────────────
        res_lbl = QtWidgets.QLabel("Pipeline Usage")
        res_lbl.setFont(heading_font)
        left_layout.addWidget(res_lbl)
        resource_frame = QtWidgets.QFrame()
        resource_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        res_layout = QtWidgets.QVBoxLayout(resource_frame)
        res_layout.setContentsMargins(4,4,4,4)
        self.cpu_bar = QtWidgets.QProgressBar()
        self.cpu_bar.setFormat("CPU: %p%")
        self.cpu_bar.setTextVisible(True)
        res_layout.addWidget(self.cpu_bar)
        self.mem_bar = QtWidgets.QProgressBar()
        self.mem_bar.setFormat("RAM: %p%")
        self.mem_bar.setTextVisible(True)
        res_layout.addWidget(self.mem_bar)
        # GPU usage bar
        self.gpu_bar = QtWidgets.QProgressBar()
        self.gpu_bar.setFormat("GPU: %p%")
        self.gpu_bar.setTextVisible(True)
        res_layout.addWidget(self.gpu_bar)
        left_layout.addWidget(resource_frame)
        left_layout.addStretch()
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        log_lbl = QtWidgets.QLabel("Log")
        log_lbl.setFont(heading_font)
        rl.addWidget(log_lbl)

        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True);
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont('Consolas', 10))
        self.log.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.log.customContextMenuRequested.connect(self.on_log_context_menu)
        rl.addWidget(self.log)

        out_lbl = QtWidgets.QLabel("Output Files")
        out_lbl.setFont(heading_font)
        rl.addWidget(out_lbl)

        # ── Filter dropdown ─────────────────────────────────────────────
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentTextChanged.connect(self.apply_output_filter)
        filter_layout.addWidget(self.filter_combo)
        rl.addLayout(filter_layout)

        # ── Output files tree with metadata ────────────────────────────
        self.out_tree = QtWidgets.QTreeWidget()
        self.out_tree.setHeaderLabels(["Name", "Size", "Modified"])
        self.out_tree.setAlternatingRowColors(True)
        self.out_tree.setSortingEnabled(True)
        self.out_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.out_tree.customContextMenuRequested.connect(self.on_output_context_menu)
        self.out_tree.itemDoubleClicked.connect(self.open_output_file)
        rl.addWidget(self.out_tree)

    #    main_splitter.addWidget(left); main_splitter.addWidget(right); main_splitter.setSizes([400,400])
        #layout = QtWidgets.QVBoxLayout(self); layout.addWidget(main_splitter)
        main_splitter.addWidget(left)
        main_splitter.addWidget(right)
        main_splitter.setSizes([400,400])

        # ── Wrap existing UI in a tab widget ─────────────────────────
        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: Analysis Pipeline
        tab1 = QtWidgets.QWidget()
        t1_layout = QtWidgets.QVBoxLayout(tab1)
        t1_layout.addWidget(main_splitter)
        self.tabs.addTab(tab1, "Analysis Pipeline")

        # Tab 2: Recording Diagnostics (placeholder)
        # Tab 2: Recording Diagnostics
        tab2 = QtWidgets.QWidget()
        t2_layout = QtWidgets.QVBoxLayout(tab2)
        # Control row: Run button + progress bar
        ctrl_row = QtWidgets.QHBoxLayout()
        # Kilosort Directory picker
        ctrl_row.addWidget(QtWidgets.QLabel("Kilosort Directory:"))
        self.diag_dir_edit = QtWidgets.QLineEdit()
        self.diag_dir_edit.setPlaceholderText("Select directory…")
        ctrl_row.addWidget(self.diag_dir_edit)
        browse_btn = QtWidgets.QPushButton("…")
        browse_btn.clicked.connect(self._select_diag_dir)
        ctrl_row.addWidget(browse_btn)
        self.diag_btn = QtWidgets.QToolButton(objectName='diagBtn')
        self.diag_btn.setText("Run Diagnostics")
        self.diag_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.diag_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.diag_btn.clicked.connect(self.run_diagnostics)
        ctrl_row.addWidget(self.diag_btn)
        self.diag_progress = QtWidgets.QProgressBar(objectName='diagProgress')
        self.diag_progress.setRange(0, 100)
        self.diag_progress.setTextVisible(False)
        ctrl_row.addWidget(self.diag_progress, 1)
        t2_layout.addLayout(ctrl_row)
        # Info display area
        '''
        info_frame = QtWidgets.QFrame(objectName='infoFrame')
        info_frame.setProperty('card', True)
        info_layout = QtWidgets.QGridLayout(info_frame)
        stats = ["Number of templates", "Number of channels"]
        self.info_labels = {}
        for col, name in enumerate(stats):
            label = QtWidgets.QLabel(f"{name}:")
            value = QtWidgets.QLabel("N/A")
            info_layout.addWidget(label, 0, col*2)
            info_layout.addWidget(value, 0, col*2+1)
            self.info_labels[name] = value
        t2_layout.addWidget(info_frame)
        '''
        # Diagnostics info: tree on left, detail on right
        diag_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # ── Left: diagnostics tree
        self.diag_tree = QtWidgets.QTreeWidget()
        self.diag_tree.setHeaderHidden(True)
        # populate root & child nodes as placeholders
        roots = ["Overall data", "Spike statistics", "Templates", "Voltage"]
        for root_name in roots:
            root = QtWidgets.QTreeWidgetItem([root_name])
            diag_split.addWidget(self.diag_tree) if False else None
            self.diag_tree.addTopLevelItem(root)
        diag_split.addWidget(self.diag_tree)
        # ── Right: stacked widget for details
        self.diag_stack = QtWidgets.QStackedWidget()
        # page 0: summary info
        info_page = QtWidgets.QWidget()
        info_form = QtWidgets.QFormLayout(info_page)
        stats = ["Number of templates", "Number of channels", "Recording duration (s)"]
        self.info_labels = {}
        for name in stats:
            lbl = QtWidgets.QLabel("N/A")
            info_form.addRow(f"{name}:", lbl)
            self.info_labels[name] = lbl
      #  self.diag_stack.addWidget(info_page)
      #  diag_split.addWidget(self.diag_stack)
      #  t2_layout.addWidget(diag_split)
        self.diag_stack.addWidget(info_page)
        # ── Page 2: Voltage diagnostics ─────────────────────────────
        voltage_page = QtWidgets.QWidget()
        voltage_layout = QtWidgets.QVBoxLayout(voltage_page)
        # slider to pick window start
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Window start (samples):"))
        self.voltage_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.voltage_slider.setEnabled(False)
        slider_row.addWidget(self.voltage_slider, 1)
        voltage_layout.addLayout(slider_row)
        # stats for the selected window
        stats_page = QtWidgets.QWidget()
        stats_form = QtWidgets.QFormLayout(stats_page)
        self.voltage_avg_label = QtWidgets.QLabel("N/A")
        self.voltage_std_label = QtWidgets.QLabel("N/A")
        stats_form.addRow("Avg Voltage:", self.voltage_avg_label)
        stats_form.addRow("Std Dev:",   self.voltage_std_label)
        voltage_layout.addWidget(stats_page)
        # full-recording average-voltage plot
        self.voltage_fig    = Figure()
        self.voltage_canvas = FigureCanvas(self.voltage_fig)
        voltage_layout.addWidget(self.voltage_canvas)
        self.diag_stack.addWidget(voltage_page)

        # ── Waveform plot page
        wave_page = QtWidgets.QWidget()
        wave_layout = QtWidgets.QVBoxLayout(wave_page)
        self.wave_fig = Figure()
        self.wave_canvas = FigureCanvas(self.wave_fig)
        wave_layout.addWidget(self.wave_canvas)
        self.diag_stack.addWidget(wave_page)
        # connect tree selection to waveform plotting
        self.diag_tree.currentItemChanged.connect(self.on_diag_tree_selection)
        diag_split.addWidget(self.diag_stack)
        t2_layout.addWidget(diag_split)
        # Plot selector + display
        plot_frame = QtWidgets.QFrame(objectName='plotFrame')
        plot_frame.setProperty('card', True)
        plot_layout = QtWidgets.QVBoxLayout(plot_frame)
        self.plot_selector = QtWidgets.QComboBox()
        plot_layout.addWidget(self.plot_selector)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = Toolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        #self.plot_area = QtWidgets.QLabel()
        #self.plot_area.setAlignment(QtCore.Qt.AlignCenter)
        #self.plot_area.setMinimumHeight(200)
        #plot_layout.addWidget(self.plot_area)
        t2_layout.addWidget(plot_frame)
        self.tabs.addTab(tab2, "Recording Diagnostics")

        # Set the main layout to host the tabs
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)
        for m in analyses:
            self.availableList.addItem(m["name"])
        self.pipelineList.model().rowsInserted.connect(self.rebuildSettingsTabs)
        self.pipelineList.model().rowsRemoved.connect(self.rebuildSettingsTabs)

    def browseBaseDir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Base Directory")
        if d:
            self.base_dir_edit.setText(d)
            self.rebuildSettingsTabs()

    def rebuildSettingsTabs(self):
        # cache current values into self.settings_cache
        for name, widgets in getattr(self, 'settings_tabs', {}).items():
            # ensure we have a dict for this analysis
            cache = self.settings_cache.setdefault(name, {})
            # look up its args in our analyses metadata
            argspec = next((m['args'] for m in analyses if m['name']==name), [])
            for param in argspec:
                # unpack name, type, (optional) pattern
                arg     = param[0]
                typ     = param[1]
                pattern = param[2] if len(param) > 2 else None 
                w = widgets[arg]
                if isinstance(w, QtWidgets.QLineEdit):
                    cache[arg] = w.text()
                elif isinstance(w, QtWidgets.QTextEdit):
                    cache[arg] = w.toPlainText()
                elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    cache[arg] = w.value()
                elif isinstance(w, QtWidgets.QCheckBox):
                    cache[arg] = w.isChecked()
                elif isinstance(w, QtWidgets.QComboBox):
                    # restore previous enum selection from cache
                    prev = cache.get(arg, "")
                    idx  = w.findText(prev)
                    if idx >= 0:
                        w.setCurrentIndex(idx)

        # now use that cache as 'old' when rebuilding
        old = self.settings_cache
        self.settings_tabs = {}; self.settingsTabs.clear()
        names = [self.pipelineList.item(i).text() for i in range(self.pipelineList.count())]
        for idx, name in enumerate(names,1):
            meta = next((m for m in analyses if m['name']==name), None)
            if not meta or not meta['args']: continue
            tab = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(tab)
            widgets = {}
            for param in meta['args']:
                # unpack name, type, (optional) pattern
                arg     = param[0]
                typ     = param[1]
                pattern = param[2] if len(param) > 2 else None 
                if typ == "int":
                    w = QtWidgets.QSpinBox(); w.setRange(-10**9, 10**9)
                    form.addRow(f"{arg}:", w)
                elif typ == "float":
                    w = QtWidgets.QDoubleSpinBox(); w.setMaximum(1e9)
                    form.addRow(f"{arg}:", w)
                elif typ == "bool":
                    w = QtWidgets.QCheckBox()
                    form.addRow(f"{arg}:", w)
                elif typ == "Path":
                    w_line = QtWidgets.QLineEdit(); btn = QtWidgets.QPushButton("…")
                    btn.clicked.connect(lambda _,l=w_line,a=arg: l.setText(QtWidgets.QFileDialog.getOpenFileName(self,f"Select {a}")[0]))
                    container = QtWidgets.QWidget(); h=QtWidgets.QHBoxLayout(container)
                    h.setContentsMargins(0,0,0,0); h.addWidget(w_line); h.addWidget(btn)
                    w = w_line; form.addRow(f"{arg}:", container)
                elif typ == "Enum":
                    w = QtWidgets.QComboBox()
                    # `pattern` is the list of enum options
                    for option in pattern:
                        w.addItem(option)
                    form.addRow(f"{arg}:", w)
                elif typ in ("FilePath", "DirPath"):
                    w_line = QtWidgets.QLineEdit()
                    btn    = QtWidgets.QPushButton("…")
                    if typ == "DirPath":
                        chooser = lambda: QtWidgets.QFileDialog.getExistingDirectory(self, f"Select {arg}")
                    else:
                        chooser = lambda: QtWidgets.QFileDialog.getOpenFileName(self, f"Select {arg}")[0]
                    btn.clicked.connect(lambda _, w=w_line, fn=chooser: w.setText(fn()))
                    container = QtWidgets.QWidget()
                    h = QtWidgets.QHBoxLayout(container)
                    h.setContentsMargins(0,0,0,0)
                    h.addWidget(w_line); h.addWidget(btn)
                    w = w_line
                    form.addRow(f"{arg}:", container)
                elif typ == "list":
                    w = QtWidgets.QTextEdit(); w.setPlaceholderText("one per line")
                    form.addRow(f"{arg}:", w)
                else:
                     w = QtWidgets.QLineEdit()
                     form.addRow(f"{arg}:", w)
                widgets[arg] = w
                # restore old value
                if name in old and arg in old[name]:
                    val = old[name][arg]
                    if isinstance(w, QtWidgets.QLineEdit):
                        w.setText(val)
                    elif isinstance(w, QtWidgets.QTextEdit):
                        w.setPlainText(val)
                    elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                        w.setValue(val)
                    elif isinstance(w, QtWidgets.QCheckBox):
                        w.setChecked(val)
            self.settings_tabs[name] = widgets
            self.settingsTabs.addTab(tab, f"{idx}. {name}")

    def run_pipeline(self):
        for name, widgets in self.settings_tabs.items():
            meta = next(m for m in analyses if m['name'] == name)
            for param in meta['args']:
                # unpack name, type, (optional) pattern
                arg     = param[0]
                typ     = param[1]
                pattern = param[2] if len(param) > 2 else None 
                if typ in ("FilePath", "DirPath") and not widgets[arg].text():
                    QtWidgets.QMessageBox.warning(self, 'Missing', f"Please set '{arg}' for '{name}'.")
                    return

        descs = [self.pipelineList.item(i).text() for i in range(self.pipelineList.count())]
        mapping = {}
        for name in descs:
            meta = next(m for m in analyses if m['name']==name)
            widgets = self.settings_tabs.get(name, {})    # default {} for no-arg analyses
            mapping[name] = self._make_runner(meta['func'], meta['args'], widgets)

        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

        print('[GUI] Starting pipeline...\n')
        self.thread = QtCore.QThread()
        self.worker = PipelineWorker(self.base_dir_edit.text(), descs, mapping)
        self.worker.moveToThread(self.thread)
        self.worker.outputFileWritten.connect(self.handle_output_file_written)

        # when pipeline finishes, refresh all file sizes & modified times
        self.worker.finished.connect(self.refresh_output_file_info)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.thread.finished.connect(lambda: self.stop_btn.setEnabled(False))
        self.thread.start()

    def validate_file_layout(self):
        """Validate FilePath inputs against provided regex patterns."""
        print("[GUI] Validating file layout...")
        for name, widgets in self.settings_tabs.items():
            meta = next(m for m in analyses if m['name'] == name)
            for entry in meta['args']:
                # only entries with a regex (3-tuple) and type FilePath
                if len(entry) < 3:
                    if len(entry) == 2 and entry[1] == "FilePath":
                        print(f"[GUI] No layout requirements specified for {name}::{entry[0]}, skipping.")
                    continue
                arg, typ, pattern = entry
                if typ != "FilePath" or not pattern:
                    continue

                val = widgets[arg].text().strip()
                if not val:
                    print(f"[GUI] [{name}] {arg}: no path provided, skipping.\n")
                    continue
                p = Path(val)
                if not p.exists():
                    print(f"[GUI] [{name}] {arg}: path does not exist, skipping.\n")
                    continue

                # only attempt regex on text files
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, 1):
                            if not re.match(pattern, line.rstrip('\n')):
                                print(
                                    f"[GUI] [{name}] {arg}: line {i} "
                                    f"does not match pattern {pattern!r}.\n"
                                )
                                break
                        else:
                            print(f"[GUI] [{name}] {arg}: all lines match.\n")
                except UnicodeDecodeError:
                    print(
                        f"[GUI] [{name}] {arg}: not a text file, "
                        "skipping regex validation.\n"
                    )
                except Exception as e:
                    print(
                        f"[GUI] [{name}] {arg}: error reading file: {e}\n"
                    )
        print("[GUI] File validation complete!")

    def _make_runner(self, func, argspec, widgets):
        def runner(base_dir):
            vals = [base_dir]
            for param in argspec:
                # unpack name, type, (optional) pattern
                arg     = param[0]
                typ     = param[1]
                pattern = param[2] if len(param) > 2 else None                
                w = widgets[arg]
                if typ == "int":
                    vals.append(w.value())
                elif typ == "float":
                    vals.append(w.value())
                elif typ == "bool":
                    vals.append(w.isChecked())
                elif typ in ("FilePath", "DirPath"):
                    vals.append(Path(w.text()))
                elif typ == "list":
                    vals.append([l.strip() for l in w.toPlainText().splitlines() if l.strip()])
                elif typ == "Enum":
                    # pull selected enum value
                    vals.append(w.currentText())
                else:
                    vals.append(w.text())
            return func(*vals)
        return runner

    def handle_output_file_written(self, fn):
        p = Path(fn)

        # human-readable size
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            size = 0

        for unit in ['B','KB','MB','GB','TB']:
            if size < 1024.0:
                human_size = f"{size:.1f}{unit}"
                break
            size /= 1024.0

        # human-readable local timestamp, e.g. "Apr 30, 2025 1:27 PM"
        dt = QDateTime.fromSecsSinceEpoch(int(p.stat().st_mtime)).toLocalTime()
        modified = dt.toString("MMM d, yyyy h:mm ap")

        # avoid duplicates
        existing = [
            self.out_tree.topLevelItem(i).data(0, QtCore.Qt.UserRole)
            for i in range(self.out_tree.topLevelItemCount())
        ]
        if fn in existing:
            return
        # create tree item
        item = QtWidgets.QTreeWidgetItem([p.name, human_size, modified])
        item.setIcon(0, QtGui.QIcon.fromTheme('text-x-generic'))
        item.setData(0, QtCore.Qt.UserRole, fn)
        ext = p.suffix.lower()
        item.setData(0, QtCore.Qt.UserRole+1, ext)
        self.out_tree.addTopLevelItem(item)
        # add new extension to filter dropdown
        if ext and self.filter_combo.findText(ext) == -1:
            self.filter_combo.addItem(ext)

    def on_log_context_menu(self, pos):
        menu = self.log.createStandardContextMenu()
        menu.addSeparator()
        clear = menu.addAction("Clear Log")
        action = menu.exec_(self.log.mapToGlobal(pos))
        if action == clear:
            self.log.clear()

    def append_log(self, txt):
        if txt.startswith("\r"):
            # strip the \r and rebuild the last line of plain text
            stripped = txt.lstrip("\r")
            cur = self.log.toPlainText()
            i = cur.rfind("\n")
            new = stripped if i < 0 else cur[:i+1] + stripped
            self.log.setPlainText(new)
            self.log.moveCursor(QtGui.QTextCursor.End)
            self.log.ensureCursorVisible()
            return


        # detect stderr vs stdout
        is_err = (self.sender() == self.stderr_stream)

        # scroll to bottom
        self.log.moveCursor(QtGui.QTextCursor.End)

        safe = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_html = safe.replace("\n", "<br/>")

        block_style = (
            "display:block; width:100%; padding:2px 4px; margin:1px 0;"
            "border-radius:3px;"
        )

        if is_err:
            bg_color = "#3d1f1f"
            fg_color = "#ff8080"
        else:
            bg_color = "#2a2a2a"
            fg_color = "#ffffff"

        self.log.insertHtml(
            f'<div style="margin:0; padding:1px 4px; background-color:{bg_color}; color:{fg_color}; width:100%;">'
            f'{safe_html}'
            f'</div>'
        )

        # ensure we stay at the bottom
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.ensureCursorVisible()

    def save_state(self):
        state = {
            'base_dir': self.base_dir_edit.text(),
            'pipeline': [self.pipelineList.item(i).text() for i in range(self.pipelineList.count())],
            'params': {}
        }
        for name, widgets in getattr(self, 'settings_tabs', {}).items():
            state['params'][name] = {}
            argspec = next((m['args'] for m in analyses if m['name']==name), [])
            for param in argspec:
                arg     = param[0]
                typ     = param[1]
                pattern = param[2] if len(param) > 2 else None 
                w = widgets[arg]
                if isinstance(w, QtWidgets.QLineEdit):
                    state['params'][name][arg] = w.text()
                elif isinstance(w, QtWidgets.QTextEdit):
                    state['params'][name][arg] = w.toPlainText()
                elif isinstance(w, QtWidgets.QSpinBox) or isinstance(w, QtWidgets.QDoubleSpinBox):
                    state['params'][name][arg] = w.value()
                elif isinstance(w, QtWidgets.QCheckBox):
                    state['params'][name][arg] = w.isChecked()
                elif isinstance(w, QtWidgets.QComboBox):
                    state['params'][name][arg] = w.currentText()
        try:
            with open(self.STATE_FILE, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        try:
            with open(self.STATE_FILE, 'rb') as f:
                state = pickle.load(f)
        except Exception:
            return
        self.base_dir_edit.setText(state.get('base_dir', ''))
        self.pipelineList.clear()
        for it in state.get('pipeline', []):
            self.pipelineList.addItem(it)
        self.rebuildSettingsTabs()
        params = state.get('params', {})
        for name, widget_dict in getattr(self, 'settings_tabs', {}).items():
            vals = params.get(name, {})
            for arg, val in vals.items():
                w = widget_dict.get(arg)
                if not w:
                    continue
                if isinstance(w, QtWidgets.QLineEdit):
                    w.setText(val)
                elif isinstance(w, QtWidgets.QTextEdit):
                    w.setPlainText(val)
                elif isinstance(w, QtWidgets.QSpinBox) or isinstance(w, QtWidgets.QDoubleSpinBox):
                    w.setValue(val)
                elif isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(val)
                elif isinstance(w, QtWidgets.QComboBox):
                    # restore enum selection
                    idx = w.findText(val)
                    if idx >= 0:
                        w.setCurrentIndex(idx)

    def stop_pipeline(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def open_output_file(self, item, column=None):
        # handle both QListWidgetItem and QTreeWidgetItem
        if isinstance(item, QtWidgets.QTreeWidgetItem):
            fn = item.data(0, QtCore.Qt.UserRole)
        else:
            fn = item.text()
        self.open_path(fn)

    def open_path(self, fn):
        p = Path(fn)
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        else:
            QtWidgets.QMessageBox.warning(self, 'File Not Found', f"Cannot find {p}")

    def apply_output_filter(self, text):
        for i in range(self.out_tree.topLevelItemCount()):
            item = self.out_tree.topLevelItem(i)
            ext = item.data(0, QtCore.Qt.UserRole+1)
            item.setHidden(False if text == "All" or ext == text else True)

    def on_output_context_menu(self, pos):
        item = self.out_tree.itemAt(pos)
        if not item:
            return
        fn = item.data(0, QtCore.Qt.UserRole)
        menu = QtWidgets.QMenu()
        open_act   = menu.addAction("Open")
        reveal_act = menu.addAction("Reveal in Explorer")
        copy_act   = menu.addAction("Copy Path")
        open_act.triggered.connect(lambda: self.open_path(fn))
        reveal_act.triggered.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(fn).parent))))
        copy_act.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(fn))
        menu.exec_(self.out_tree.viewport().mapToGlobal(pos))

    def refresh_output_file_info(self):
        for i in range(self.out_tree.topLevelItemCount()):
            item = self.out_tree.topLevelItem(i)
            fn = item.data(0, QtCore.Qt.UserRole)
            p = Path(fn)
            if not p.exists():
                continue
            # size
            size = p.stat().st_size
            for unit in ['B','KB','MB','GB','TB']:
                if size < 1024.0:
                    human = f"{size:.1f}{unit}"
                    break
                size /= 1024.0
            item.setText(1, human)
            # modified
            dt = QDateTime.fromSecsSinceEpoch(int(p.stat().st_mtime)).toLocalTime()
            item.setText(2, dt.toString("MMM d, yyyy h:mm ap"))

    def closeEvent(self, e):
        self.save_state()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        e.accept()

    def _select_diag_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Kilosort Directory",  
            self.base_dir_edit.text() or QtCore.QDir.homePath()
        )
        if d:
            self.diag_dir_edit.setText(d)
            self.diagnostics_directory = Path(d)

    def run_diagnostics(self):
        print(f"Loading diagnostics data from {self.diagnostics_directory}")
        # load data from Kilosort directory
        templates = np.load(self.diagnostics_directory / "templates.npy")
        self.templates = templates
        # reset progress
        self.diag_progress.setValue(0)

        # placeholder loop to simulate work
        for i in range(101):
            QtCore.QCoreApplication.processEvents()
            self.diag_progress.setValue(i)

        # simulate populating stats
        self.info_labels["Number of templates"].setText(f"{templates.shape[0]}")
        self.info_labels["Number of channels"].setText(f"{templates.shape[2]}")
        for i in range(self.diag_tree.topLevelItemCount()):
            node = self.diag_tree.topLevelItem(i)
            if node.text(0) == "Templates":
                node.takeChildren()
                for j in range(templates.shape[0]):
                    QtWidgets.QTreeWidgetItem(node, [f"Template {j}"])
                break
        # simulate populating stats
        self.info_labels["Number of templates"].setText(f"{templates.shape[0]}")
        self.info_labels["Number of channels"].setText(f"{templates.shape[2]}")
        # initialize plot selector
        self.plot_selector.clear()
        self.plot_selector.addItems(["Drift over time"])
        self.plot_selector.currentIndexChanged.connect(self.update_diagnostic_plot)
        self.update_diagnostic_plot(self.plot_selector.currentIndex())

        # ── Load raw binary and prepare voltage diagnostics ─────────────
        ops = np.load(self.diagnostics_directory / 'ops.npy', allow_pickle=True).item()
        raw_file = Path(ops['data_dir']) / ops['filename']
        print(f"Loading voltage data from {raw_file}")
        with open(raw_file, "rb") as fid:
            raw = np.fromfile(fid, dtype=np.int16)
        num_ch = templates.shape[2] + 1
        samples = raw.size // num_ch
        data = raw.reshape((samples, num_ch), order="F")
        self.raw_data = data

        # compute and cache per-window per-channel stats
        window = 2 * 60 * 30000
        starts = np.arange(0, data.shape[0] - window + 1, window)
        self.voltage_avgs_pc = np.array([data[i:i+window, :].mean(axis=0) for i in starts])
        self.voltage_stds_pc = np.array([data[i:i+window, :].std(axis=0)  for i in starts])

        # slider now indexes precomputed windows
        self.voltage_slider.setRange(0, len(starts) - 1)
        self.voltage_slider.setEnabled(True)
        self.voltage_slider.valueChanged.connect(self.update_voltage_window)

        # plot full-recording average voltage per channel
        times = starts / 30000
        self.voltage_fig.clear()
        ax = self.voltage_fig.add_subplot(111)
        ax.plot(times, self.voltage_avgs_pc.mean(axis=1))
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Avg Voltage")
        # (If too many channels, you may disable legend to reduce clutter)
        self.voltage_canvas.draw()

        # display stats for the first window
        self.update_voltage_window(0)

    def update_diagnostic_plot(self, index):
        #title = self.plot_selector.currentText()

        # TODO: replace with actual plot rendering
        #self.plot_area.setText(f"[Preview of '{title}' will appear here]")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        title = self.plot_selector.currentText()
        ax.set_title(title)
        # dummy data for illustration
        if title == "Drift over time":
            ops = np.load(self.diagnostics_directory / "ops.npy", allow_pickle=True).item()
            dshift = ops['dshift']
            Nbatches = ops.get('Nbatches', len(dshift))
            time = np.arange(Nbatches) * 2  # Assuming a batch represents 2 sec.
            ax.plot(time, dshift)
            ax.set_xlabel('Time (sec.)')
            ax.set_ylabel('Drift (um)')
            ax.set_title(f"Drift for {self.diagnostics_directory}")
        else:
            ax.plot(np.random.randn(100).cumsum())
        self.canvas.draw()

    def on_diag_tree_selection(self, current, previous):
        """Switch diagnostics tab based on tree selection."""
        if current is None:
            return
        # Voltage root: show voltage page
        if current.text(0) == "Voltage":
            self.diag_stack.setCurrentIndex(1)
            return
        # Templates child: waveform page
        parent = current.parent()
        if parent and parent.text(0) == "Templates":
            try:
                idx = int(current.text(0).split()[-1])
            except ValueError:
                return
            tpl = self.templates[idx]
            self.wave_fig.clear()
            ax = self.wave_fig.add_subplot(111)
            for ch in range(tpl.shape[1]):
                data = tpl[:, ch]
                if np.any(data):
                    ax.plot(data, label=f"Ch {ch}")
            ax.legend(loc='upper right')
            ax.set_title(f"Template {idx} Waveforms")
            self.wave_canvas.draw()
            self.diag_stack.setCurrentIndex(self.diag_stack.count()-1)
            return
        # all other clicks: summary info page
        self.diag_stack.setCurrentIndex(0)

    def update_voltage_window(self, idx):
        """Display cached statistics for window index idx."""
        avg_vec = self.voltage_avgs_pc[idx]
        std_vec = self.voltage_stds_pc[idx]
        # show overall mean & std across channels
        self.voltage_avg_label.setText(f"{avg_vec.mean():.3f}")
        self.voltage_std_label.setText(f"{std_vec.mean():.3f}")

    def save_state_as(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Pipeline State As",
            str(Path.home()), "Pipeline State (*.pkl)")
        if not fname:
            return
        state = {
            'base_dir': self.base_dir_edit.text(),
            'pipeline': [self.pipelineList.item(i).text()
                         for i in range(self.pipelineList.count())],
            'params': {}
        }
        for name, widgets in getattr(self, 'settings_tabs', {}).items():
            state['params'][name] = {}
            argspec = next((m['args'] for m in analyses
                            if m['name']==name), [])
            for arg, _ in argspec:
                w = widgets[arg]
                if isinstance(w, QtWidgets.QLineEdit):
                    state['params'][name][arg] = w.text()
                elif isinstance(w, QtWidgets.QTextEdit):
                    state['params'][name][arg] = w.toPlainText()
                elif isinstance(w, (QtWidgets.QSpinBox,
                                    QtWidgets.QDoubleSpinBox)):
                    state['params'][name][arg] = w.value()
                elif isinstance(w, QtWidgets.QCheckBox):
                    state['params'][name][arg] = w.isChecked()
                elif isinstance(w, QtWidgets.QComboBox):
                    state['params'][name][arg] = w.currentText()
        try:
            with open(fname, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Could not save state:\n{e}")

    def load_state_as(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Pipeline State",
            str(Path.home()), "Pipeline State (*.pkl)")
        if not fname:
            return
        try:
            with open(fname, 'rb') as f:
                state = pickle.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Could not load state:\n{e}")
            return
        self.base_dir_edit.setText(state.get('base_dir',''))
        self.pipelineList.clear()
        for it in state.get('pipeline', []):
            self.pipelineList.addItem(it)
        self.rebuildSettingsTabs()
        params = state.get('params', {})
        for name, widget_dict in getattr(self, 'settings_tabs', {}).items():
            vals = params.get(name, {})
            for arg, val in vals.items():
                w = widget_dict.get(arg)
                if not w:
                    continue
                if isinstance(w, QtWidgets.QLineEdit):
                    w.setText(val)
                elif isinstance(w, QtWidgets.QTextEdit):
                    w.setPlainText(val)
                elif isinstance(w, (QtWidgets.QSpinBox,
                                    QtWidgets.QDoubleSpinBox)):
                    w.setValue(val)
                elif isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(val)
                elif isinstance(w, QtWidgets.QComboBox):
                    idx = w.findText(val)
                    if idx >= 0:
                        w.setCurrentIndex(idx)

    def update_resources(self):
        # ── Per-process CPU & RAM usage ─────────────────────────────
        procs = [self._proc] + self._proc.children(recursive=True)
        total_cpu = sum(p.cpu_percent(None) for p in procs)
        total_cpu = min(total_cpu / psutil.cpu_count(), 100.0)
        total_mem = sum(p.memory_percent() for p in procs)
        self.cpu_bar.setValue(int(total_cpu))
        self.mem_bar.setValue(int(total_mem))

         # ── Per-process GPU memory usage ────────────────────────────
        used = 0
        for h in self._gpu_handles:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except NVMLError_FunctionNotFound:
                procs = []
            for proc_info in procs:
                if proc_info.pid == os.getpid():
                    used += proc_info.usedGpuMemory
        # If NVML compute-process API unavailable, GPU usage will be 0
        percent = min(100, (used / sum(self._gpu_total_mem.values())) * 100) if used else 0
        self.gpu_bar.setValue(int(percent))

if __name__ == '__main__':
    # ── Make app DPI-aware for fullscreen ────────────────────────────
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)

    # ── Global styling ────────────────────────────
    font = app.font()               
    font.setFamily("Segoe UI")         
    font.setPointSize(14)             
    app.setFont(font)                 
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # ── Set to fullscreen by default and then enter ────────────
    win = MainWindow();
    screen_rect = app.primaryScreen().availableGeometry()
    win.setGeometry(screen_rect)   # stretch to fill
    win.show() 
    sys.exit(app.exec_())
