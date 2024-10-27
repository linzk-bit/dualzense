from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QVBoxLayout
from PyQt5.QtWidgets import QTableWidgetItem
from mainwindow_ui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from eisdata import EISDataSet, EISBattery, BatteryInfo
import statsmodels.api as sm
from scipy.optimize import curve_fit

_FREQ_DICT = {'1': 100, '2': 50, '3': 10, '4': 1}
_GAP_DICT = {'1': 3.16, '2': 5.01, '3': 10, '4': 0}
_TYPE_DICT = {'0': 'real', '1': 'imag', '2': 'norm', '3': 'phase'}


class EISTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.connect_slots()
        self.add_preview_canvas()
        self.add_sensitive_canvas()
        self.add_fit_canvas()
        self._data = None

    def connect_slots(self):
        # data panel
        self.ui.btn_add_data.clicked.connect(self.add_data)
        self.ui.btn_clear_data.clicked.connect(self.clear_data)
        self.ui.btn_del_row.clicked.connect(self.delete_row)
        self.ui.btn_export_pkl.clicked.connect(self.export_pkl)
        self.ui.col_freq.valueChanged.connect(self.change_col)
        # preview panel
        self.ui.btn_plot_EIS.clicked.connect(self.plot_EIS)
        # map panel
        self.ui.btn_plot_map.clicked.connect(self.plot_map)
        self.ui.btn_export_map.clicked.connect(self.export_map)
        # fit panel
        self.ui.btn_fit.clicked.connect(self.fit)
        self.ui.btn_export_fit.clicked.connect(self.export_fit)
        # analyze panel
        self.ui.btn_add_analyze.clicked.connect(self.add_analyze)
        self.ui.btn_clear_analyze.clicked.connect(self.clear_analyze)
        self.ui.btn_del_analyze.clicked.connect(self.delete_analyze)
        self.ui.btn_run_analyze.clicked.connect(self.run_analyze)
        self.ui.btn_export_analyze.clicked.connect(self.export_analyze)
        self.ui.btn_set_default_ref.clicked.connect(self.set_default_ref)
        self.ui.btn_import_fit_param.clicked.connect(self.import_fit_param)

    def add_preview_canvas(self):
        self.preview_fig = Figure()
        self.ax_preview = self.preview_fig.add_subplot(111, projection='3d')
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.layout = QVBoxLayout(self.ui.previewer)
        self.layout.addWidget(self.preview_canvas)
        self.ax_preview.set_ylabel('T (℃)')
        self.ax_preview.set_xlabel("Z'")
        self.ax_preview.set_zlabel("-Z''")
        self.ax_preview.set_title('EIS')

    def add_sensitive_canvas(self):
        self.sensitive_fig = Figure()
        self.ax_sensitive = self.sensitive_fig.add_subplot(111)
        self.sensitive_canvas = FigureCanvas(self.sensitive_fig)
        self.layout = QVBoxLayout(self.ui.sensitivity)
        self.layout.addWidget(self.sensitive_canvas)
        self.ax_sensitive.set_xlabel("T (℃)")
        self.ax_sensitive.set_ylabel("Freq (Hz)")
        self.ax_sensitive.set_title(
            f'Sensitivity of {self.ui.cbx_map_type.currentText()[6:]}')

    def add_fit_canvas(self):
        self.fit_fig = Figure()
        self.ax_fit = self.fit_fig.add_subplot(111)
        self.fit_canvas = FigureCanvas(self.fit_fig)
        self.layout = QVBoxLayout(self.ui.fitting)
        self.layout.addWidget(self.fit_canvas)
        self.ax_fit.set_xlabel("T (℃)")
        self.ax_fit.set_ylabel("Fit Value")
        self.ax_fit.set_title(
            f'Fit of {self.ui.cbx_fit_type.currentText()[6:]}')

    def _data_append_pkl(self, pkl_file: str):
        if pkl_file.endswith('.pkl'):
            if self._data is None:
                self._data = EISDataSet(
                    soc=[], soh=[], info=BatteryInfo(), eis=EISBattery([], [], [], []))
            new_dataset = EISDataSet.from_pkl(pkl_file)
            self._data.extend(new_dataset)
        else:
            raise Exception('Only support pkl files')

    def _analyze_data_append_pkl(self, pkl_file: str):
        if pkl_file.endswith('.pkl'):
            if self._analyze_data is None:
                self._analyze_data = EISDataSet(
                    soc=[], soh=[], info=BatteryInfo(), eis=EISBattery([], [], [], []))
            new_dataset = EISDataSet.from_pkl(pkl_file)
            self._analyze_data.extend(new_dataset)
        else:
            raise Exception('Only support pkl files')

    def _data_append(self, temperature, frequency, real, imag, soc=None, soh=None, info: dict = {}):
        if self._data is None:
            self._data = EISDataSet(
                soc=[], soh=[], info=BatteryInfo(), eis=EISBattery([], [], [], []))
        new_dataset = EISDataSet(soc, soh, BatteryInfo(
            **info), EISBattery([temperature], [frequency], [real], [imag]))
        self._data.extend(new_dataset)

    def _pick_data_from_file(self, file):
        skiprows = self.ui.skip_rows.value()

        if file.split('.')[-1].lower() in ['csv', 'txt']:
            delimiters = [None, ',', ';']
            for d in delimiters:
                try:
                    data = np.loadtxt(file, delimiter=d, skiprows=skiprows)
                    self.ui.statusbar.showMessage(
                        f'load data success d={d},skip={skiprows}.')
                    break
                except Exception as e:
                    self.ui.statusbar.showMessage(
                        f'load data failed d={d},skip={skiprows}: {e}')

        elif file.split('.')[-1].lower() == 'xlsx':
            data = pd.read_excel(file)
            data = data.to_numpy()

        else:
            raise Exception(f'Unsupported file type: {file.split("/")[-1]}')

        if data is None:
            raise Exception(f'Failed to load data from {file.split("/")[-1]}.')

        try:
            f_col = self.ui.col_freq.value() - 1
            re_col = self.ui.col_real.value() - 1
            im_col = self.ui.col_imag.value() - 1
            f = data[:, f_col]
            re = data[:, re_col]
            neg_im = data[:, im_col]
            if not self.ui.is_neg_imag.isChecked():
                neg_im = -neg_im
            return f, re, neg_im

        except Exception as e:
            self.ui.statusbar.showMessage(
                f'Data format error: {file.split("/")[-1]}')
            return None

    def _load_data(self):
        self._data = EISDataSet(
            soc=[], soh=[], info=BatteryInfo(), eis=EISBattery([], [], [], []))
        file_list = [self.ui.table.item(i, 1).text()
                     for i in range(self.ui.table.rowCount())]
        temperature = []
        frequency = []
        real = []
        neg_imag = []
        for i, file in enumerate(file_list):
            if file.split('.')[-1].lower() not in ['csv', 'txt', 'xlsx']:
                continue
            T = float(self.ui.table.item(i, 0).text())
            try:
                pdata = self._pick_data_from_file(file)
                f, re, neg_im = pdata
            except Exception as e:
                self.ui.statusbar.showMessage(
                    f'Load data failed: {file.split("/")[-1]}. Error: {e}')
                f = None
                re = None
                neg_im = None
            temperature.append(T)
            frequency.append(f)
            real.append(re)
            neg_imag.append(neg_im)
        if len(temperature) > 0:
            self._data_append(temperature, frequency, real,
                              neg_imag, soc=None, soh=None, info={})

        for i, file in enumerate(file_list):
            if file.split('.')[-1].lower() == 'pkl':
                self._data_append_pkl(file)
        self.ui.statusbar.showMessage('All files are loaded.')

    def _get_delta_from_eis(self, frequency, real, neg_imag, f0, f1, type=0):
        idx_f0 = np.argmin(np.abs(np.log(frequency)-np.log(f0)))
        idx_f1 = np.argmin(np.abs(np.log(frequency)-np.log(f1)))
        match type:
            case 0:
                delta = real[idx_f0] - real[idx_f1]
            case 1:
                delta = neg_imag[idx_f0] - neg_imag[idx_f1]
            case 2:
                delta = np.sqrt(real[idx_f0]**2 + neg_imag[idx_f0]**2) - \
                    np.sqrt(real[idx_f1]**2 + neg_imag[idx_f1]**2)
            case 3:
                delta = np.arctan2(
                    neg_imag[idx_f0], real[idx_f0]) - np.arctan2(neg_imag[idx_f1], real[idx_f1])
                delta = np.rad2deg(delta)
            case _:
                raise Exception('Invalid type')
        return delta

    def _t_from_y(self, y, A, B, C):
        d = B**2 - 4*A*(C-y)
        x = (-B + np.sqrt(d)) / (2*A)
        return 1/x

    def _run_analyze_from_row(self, row):
        file = self.ui.table_analyze.item(row, 0).text()
        delta_ref = float(self.ui.table_analyze.item(row, 2).text())
        T_ref = float(self.ui.table_analyze.item(row, 3).text())
        f0 = self.ui.cbx_analyze_f0.currentIndex()
        gap = self.ui.cbx_analyze_freq_gap.currentIndex()
        f0 = _FREQ_DICT[str(f0)]
        f1 = f0 * _GAP_DICT[str(gap)]
        analyze_type = self.ui.cbx_analyze_type.currentIndex()
        A = self.ui.val_A.value()
        B = self.ui.val_B.value()
        C = self.ui.val_C.value()
        delta30 = delta_ref / (1 + np.poly1d([A, B, C])(1/T_ref))

        if file.split('.')[-1].lower() == 'pkl':
            data = EISDataSet.from_pkl(file)
            Δ = []
            y = []
            T_pred = []
            for i, fs in enumerate(data.eis.frequency):
                for j, f in enumerate(fs):
                    re = data.eis.real[i][j]
                    neg_im = data.eis.neg_imag[i][j]
                    delta = self._get_delta_from_eis(
                        f, re, neg_im, f0, f1, type=analyze_type)
                    yij = (delta - delta30) / delta30
                    y.append(yij)
                    T_pred.append(self._t_from_y(yij, A, B, C))
                    Δ.append(delta)
            T_str = ','.join([f'{t:.2f}' for t in T_pred])
            self.ui.table_analyze.setItem(row, 1, QTableWidgetItem(T_str))
            return Δ, y, T_pred
        if file.split('.')[-1].lower() in ['csv', 'txt', 'xlsx']:
            data = self._pick_data_from_file(file)
            if data is None:
                self.ui.statusbar.showMessage('File format is not supported')
                return None
            else:
                f, re, neg_im = data
                delta = self._get_delta_from_eis(
                    f, re, neg_im, f0, f1, type=analyze_type)
                y = (delta - delta30) / delta30
                T = self._t_from_y(y, A, B, C)
                self.ui.table_analyze.setItem(
                    row, 1, QTableWidgetItem(f'{T:.2f}'))
                return delta, y, T

    def _run_analyze(self):
        row_to_analyze = self.ui.table_analyze.rowCount()
        if row_to_analyze == 0:
            self.ui.statusbar.showMessage('No data to analyze')
            return

        Δ, Y, T_pred = [], [], []

        for i in range(self.ui.table_analyze.rowCount()):
            delta, y, t = self._run_analyze_from_row(i)
            if isinstance(delta, list):
                Δ.extend(delta)
                Y.extend(y)
                T_pred.extend(t)
            else:
                Δ.append(delta)
                Y.append(y)
                T_pred.append(t)
        return Δ, Y, T_pred

    def change_col(self):
        col = self.ui.col_freq.value()
        self.ui.col_real.setValue(col+1)
        self.ui.col_imag.setValue(col+2)

    # Slots
    # import data buttons

    def add_data(self):
        fileNames, _ = QFileDialog.getOpenFileNames(
            None, "Select File", "", "data Files (*.csv;*.txt;*.pkl;*.xlsx);;All Files (*)")
        if not fileNames:
            return
        for fileName in fileNames:
            row = self.ui.table.rowCount()
            self.ui.table.insertRow(row)
            self.ui.table.setItem(row, 1, QTableWidgetItem(fileName))
            try:
                file = Path(fileName)
                t = float(file.stem)
                if -273 <= t <= 1000:
                    self.ui.table.setItem(row, 0, QTableWidgetItem(f"{t:.2f}"))
                else:
                    self.ui.table.setItem(row, 0, QTableWidgetItem("0"))
            except:
                self.ui.table.setItem(row, 0, QTableWidgetItem("0"))
            self.ui.table.resizeColumnsToContents()
        self.ui.btn_plot_EIS.setEnabled(True)
        self.ui.btn_plot_map.setEnabled(True)
        self.ui.btn_fit.setEnabled(True)
        self.ui.btn_export_pkl.setEnabled(True)

    def clear_data(self):
        self.ui.table.setRowCount(0)
        self.ui.btn_plot_EIS.setEnabled(False)
        self.ui.btn_plot_map.setEnabled(False)
        self.ui.btn_fit.setEnabled(False)
        self.ui.btn_export_map.setEnabled(False)
        self.ui.btn_export_fit.setEnabled(False)
        self.ui.btn_export_pkl.setEnabled(False)

    def delete_row(self):
        selected_rows = [index.row()
                         for index in self.ui.table.selectedIndexes()]
        rows_to_remove = sorted(list(set(selected_rows)), reverse=True)
        for row in rows_to_remove:
            self.ui.tableWidget.removeRow(row)
        row = self.ui.table.rowCount()
        if row == 0:
            self.ui.btn_plot_EIS.setEnabled(False)
            self.ui.btn_plot_map.setEnabled(False)
            self.ui.btn_fit.setEnabled(False)
            self.ui.btn_export_map.setEnabled(False)
            self.ui.btn_export_fit.setEnabled(False)
            self.ui.btn_export_pkl.setEnabled(False)

    def export_pkl(self):
        try:
            fileName, _ = QFileDialog.getSaveFileName(
                None, "Save File", "", "Pickle Files (*.pkl);;All Files (*)")
            if not fileName:
                return
            try:
                self._load_data()
            except Exception as e:
                self.ui.statusbar.showMessage('load: '+str(e))
                return
            self._data.to_pkl(fileName)
        except Exception as e:
            self.ui.statusbar.showMessage('save: '+fileName+str(e))

    # preview pannel
    def plot_EIS(self):
        try:
            self.ax_preview.clear()
            self._load_data()
            eis = self._data.eis
            temperature = eis.temperature
            freq = eis.frequency
            re = eis.real
            im = eis.neg_imag
            preview_type = self.ui.preview_type.currentIndex()

            for i in range(len(temperature)):
                for j in range(len(temperature[i])):
                    match preview_type:
                        case 0:
                            x = re[i][j]
                            z = im[i][j]
                        case 1:
                            x = np.log10(freq[i][j])
                            z = np.sqrt(re[i][j]**2 + im[i][j]**2)
                        case 2:
                            x = np.log10(freq[i][j])
                            z = np.rad2deg(np.arctan2(im[i][j], re[i][j]))
                        case 3:
                            x = np.log10(freq[i][j])
                            z = re[i][j]
                        case 4:
                            x = np.log10(freq[i][j])
                            z = im[i][j]
                    y = temperature[i][j]*np.ones(x.shape)
                    self.ax_preview.plot(x, y, z, 'o--', alpha=0.5)
            
            self.ax_preview.set_ylabel("T (℃)")
            if preview_type == 0:
                self.ax_preview.set_xlabel("Z'")
                self.ax_preview.set_zlabel("-Z''")
            elif preview_type == 1:
                self.ax_preview.set_xlabel("lg(f/Hz)")
                self.ax_preview.set_zlabel("|Z|")

            elif preview_type == 2:
                self.ax_preview.set_xlabel("lg(f/Hz)")
                self.ax_preview.set_zlabel("Phase (°)")
            elif preview_type == 3:
                self.ax_preview.set_xlabel("lg(f/Hz)")
                self.ax_preview.set_zlabel("Z'")
            elif preview_type == 4:
                self.ax_preview.set_xlabel("lg(f/Hz)")
                self.ax_preview.set_zlabel("-Z''")
            self.preview_canvas.draw()

        except Exception as e:
            self.ui.statusbar.showMessage('plot_EIS: ' + str(e))

    def plot_map(self):
        try:
            self._load_data()
        except Exception as e:
            self.ui.statusbar.showMessage('load: '+str(e))
            return

        # sensitivity map data require temperature and frequency
        try:
            temperature = np.vstack(self._data.eis.temperature[0])
            f = np.vstack(self._data.eis.frequency[0])
            re = np.vstack(self._data.eis.real[0])
            im = np.vstack(self._data.eis.neg_imag[0])
        except Exception as e:
            self.ui.statusbar.showMessage(
                'length of data is not consist: '+str(e))
            return

        gap = self.ui.cbx_freq_gap.currentIndex()
        try:
            gap_dict = {'1': 3.16, '2': 5.01, '3': 10, '4': 0}
            gap = gap_dict[str(gap)]
        except KeyError:
            self.ui.statusbar.showMessage('gap:'+str(gap))
            return
        f0 = f[0, :]
        if gap > 0:
            num_gap = np.argmin(
                f0) - np.argmin(np.abs(np.log(f0) - np.log(gap*f0.min())))
            self.ui.statusbar.showMessage('num_gap:'+str(num_gap))
            if num_gap < 0 or num_gap > len(f0)-1:
                self.ui.statusbar.showMessage('gap out of bound.')
                return
        else:
            num_gap = 0

        map_type = self.ui.cbx_map_type.currentIndex()
        map_func = self.ui.cbx_map_func.currentText().lower()
        # using Z'
        if map_type == 0:
            if num_gap == 0:
                Δ = re
                y, x = np.meshgrid(f0, temperature)
                vmax = Δ.max()
                vmin = Δ.min()
                levels = np.linspace(vmin, vmax, 20)
            else:
                Δ = re[:, num_gap:] - re[:, :-num_gap]
                y, x = np.meshgrid(f0[num_gap:], temperature)
                vmax = np.abs(Δ).max()
                vmin = -vmax
                levels = np.linspace(vmin, vmax, 20)
            try:
                self.ax_sensitive.clear()
                if map_func == 'heatmap':
                    self.ax_sensitive.pcolor(
                        x, y, Δ, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                elif map_func == 'contourf':
                    self.ax_sensitive.contourf(
                        x, y, Δ, levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                else:
                    self.ui.statusbar.showMessage(
                        'map_function unknow: '+str(map_func))
                if self.ui.is_show_contour.isChecked():
                    c = self.ax_sensitive.contour(
                        x, y, Δ, levels=levels, colors='k', alpha=0.5)
                    self.ax_sensitive.clabel(c, inline=True, fontsize=8)
                self.ax_sensitive.set_yscale('log')
                self.ax_sensitive.set_xlabel("T (℃)")
                if num_gap == 0:
                    self.ax_sensitive.set_ylabel(r"$Z'$ (ohm)")
                else:
                    self.ax_sensitive.set_ylabel(r"$\Delta Z'$ (ohm)")
                self.sensitive_canvas.draw()
            except Exception as e:
                self.ui.statusbar.showMessage(str(e))
                return

        # using -Z''
        elif map_type == 1:
            if num_gap == 0:
                Δ = im
                y, x = np.meshgrid(f0, temperature)
                vmax = Δ.max()
                vmin = Δ.min()
                levels = np.linspace(vmin, vmax, 20)
            else:
                Δ = im[:, num_gap:] - im[:, :-num_gap]
                y, x = np.meshgrid(f0[num_gap:], temperature)
                vmax = np.abs(Δ).max()
                vmin = -vmax
                levels = np.linspace(vmin, vmax, 20)
            try:
                self.ax_sensitive.clear()
                if map_func == 'heatmap':
                    self.ax_sensitive.pcolor(
                        x, y, Δ, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                elif map_func == 'contourf':
                    self.ax_sensitive.contourf(
                        x, y, Δ, levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                else:
                    self.ui.statusbar.showMessage(
                        'map_function unknow: '+str(map_func))
                if self.ui.is_show_contour.isChecked():
                    c = self.ax_sensitive.contour(
                        x, y, Δ, levels=levels, colors='k', alpha=0.5)
                    self.ax_sensitive.clabel(c, inline=True, fontsize=8)
                self.ax_sensitive.set_yscale('log')
                self.ax_sensitive.set_xlabel("T (℃)")
                if num_gap == 0:
                    self.ax_sensitive.set_ylabel(r"$-Z''$ (ohm)")
                else:
                    self.ax_sensitive.set_ylabel(r"$\Delta (-Z'')$ (ohm)")
                self.sensitive_canvas.draw()
            except Exception as e:
                self.ui.statusbar.showMessage(str(e))
                return

        # using |Z|
        elif map_type == 2:
            Z = np.sqrt(re**2 + im**2)
            if num_gap == 0:
                Δ = Z
                y, x = np.meshgrid(f0, temperature)
                vmax = Δ.max()
                vmin = Δ.min()
                levels = np.linspace(vmin, vmax, 20)
            else:
                Δ = Z[:, num_gap:] - Z[:, :-num_gap]
                y, x = np.meshgrid(f0[num_gap:], temperature)
                vmax = np.abs(Δ).max()
                vmin = -vmax
                levels = np.linspace(vmin, vmax, 20)
            try:
                self.ax_sensitive.clear()
                if map_func == 'heatmap':
                    self.ax_sensitive.pcolor(
                        x, y, Δ, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                elif map_func == 'contourf':
                    self.ax_sensitive.contourf(
                        x, y, Δ, levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                else:
                    self.ui.statusbar.showMessage(
                        'map_function unknow: '+str(map_func))
                if self.ui.is_show_contour.isChecked():
                    c = self.ax_sensitive.contour(
                        x, y, Δ, levels=levels, colors='k', alpha=0.5)
                    self.ax_sensitive.clabel(c, inline=True, fontsize=8)
                self.ax_sensitive.set_yscale('log')
                self.ax_sensitive.set_xlabel("T (℃)")
                if num_gap == 0:
                    self.ax_sensitive.set_ylabel(r"$|Z|$ (ohm)")
                else:
                    self.ax_sensitive.set_ylabel(r"$\Delta (|Z|)$ (ohm)")
                self.sensitive_canvas.draw()
            except Exception as e:
                self.ui.statusbar.showMessage(str(e))
                return

        # using phase
        elif map_type == 3:
            phase = np.arctan2(im, re)
            if num_gap == 0:
                Δ = phase
                y, x = np.meshgrid(f0, temperature)
                vmax = Δ.max()
                vmin = Δ.min()
                levels = np.linspace(vmin, vmax, 20)
            else:
                Δ = phase[:, num_gap:] - phase[:, :-num_gap]
                y, x = np.meshgrid(f0[num_gap:], temperature)
                vmax = np.abs(Δ).max()
                vmin = -vmax
                levels = np.linspace(vmin, vmax, 20)
            try:
                self.ax_sensitive.clear()
                if map_func == 'heatmap':
                    self.ax_sensitive.pcolor(
                        x, y, Δ, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                elif map_func == 'contourf':
                    self.ax_sensitive.contourf(
                        x, y, Δ, levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                else:
                    self.ui.statusbar.showMessage(
                        'map_function unknow: '+str(map_func))
                if self.ui.is_show_contour.isChecked():
                    c = self.ax_sensitive.contour(
                        x, y, Δ, levels=levels, colors='k', alpha=0.5)
                    self.ax_sensitive.clabel(c, inline=True, fontsize=8)
                self.ax_sensitive.set_yscale('log')
                self.ax_sensitive.set_xlabel("T (℃)")
                if num_gap == 0:
                    self.ax_sensitive.set_ylabel(r"Phase (degree)")
                else:
                    self.ax_sensitive.set_ylabel(r"Phase (degree)")
                self.sensitive_canvas.draw()
            except Exception as e:
                self.ui.statusbar.showMessage(str(e))
                return
        self._map_data = [x, y, Δ]
        self.ui.btn_export_map.setEnabled(True)

    def export_map(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(
                None, "Save File", "", "Text Files (*.txt)")
            if not filename:
                return
            x, y, Δ = self._map_data
            x = x.flatten()
            y = y.flatten()
            Δ = Δ.flatten()
            data = np.column_stack([x, y, Δ])
            np.savetxt(filename, data, delimiter='\t', fmt='%.6g',
                       header='Temperature\tfrequency\tdata')

        except Exception as e:
            self.ui.statusbar.showMessage(str(e))
            return

    # fit pannel

    def fit(self):
        f0 = self.ui.cbx_fit_base_f.currentIndex()
        try:
            freq_dict = {'1': 100, '2': 50, '3': 10, '4': 1}
            f0 = freq_dict[str(f0)]
        except KeyError:
            self.ui.statusbar.showMessage('f0:'+str(f0))
            return

        gap = self.ui.cbx_fit_freq_gap.currentIndex()
        try:
            gap_dict = {'1': 3.16, '2': 5.01, '3': 10, '4': 0}
            gap = gap_dict[str(gap)]
        except KeyError:
            self.ui.statusbar.showMessage('gap:'+str(gap))
            return

        fit_type = self.ui.cbx_fit_type.currentIndex()

        try:
            self._load_data()
            eis = self._data.eis
            Δ = []
            T = []
            Δref = []
            y = []
            for i in range(len(eis.temperature)):
                Δi = []
                Ti = []
                for j, t in enumerate(eis.temperature[i]):
                    f = eis.frequency[i][j]
                    re = eis.real[i][j]
                    im = eis.neg_imag[i][j]
                    if f.min() <= f0 and f.max() >= f0 * gap:
                        idx_f0 = np.argmin(np.abs(np.log(f) - np.log(f0)))
                        idx_f1 = np.argmin(
                            np.abs(np.log(f) - np.log(f0 * gap)))
                    else:
                        continue
                    if fit_type == 0:
                        delta = re[idx_f0] - re[idx_f1]
                    elif fit_type == 1:
                        delta = im[idx_f0] - im[idx_f1]
                    elif fit_type == 2:
                        delta = np.sqrt(
                            re[idx_f0]**2 + im[idx_f0]**2) - np.sqrt(re[idx_f1]**2 + im[idx_f1]**2)
                    elif fit_type == 3:
                        delta = np.arctan2(
                            im[idx_f0], re[idx_f0]) - np.arctan2(im[idx_f1], re[idx_f1])
                    else:
                        self.ui.statusbar.showMessage(
                            'fit type unknow: '+str(fit_type))
                        return
                    Δi.append(delta)
                    Ti.append(t)
                Δi = np.array(Δi).flatten()
                Ti = np.array(Ti).flatten()
                Δ.append(Δi)
                T.append(Ti)
                p = np.polyfit(1/Ti, Δi, 2)
                delta_ref = np.poly1d(p)(1/30)
                Δref.append(delta_ref)
                y.append(Δi/delta_ref - 1)
        except Exception as e:
            self.ui.statusbar.showMessage('fit: '+str(e))
            return

        try:
            self.ax_fit.clear()
            for i in range(len(y)):
                self.ax_fit.plot(T[i], y[i], '--o', alpha=0.2)
            self.ax_fit.set_xlabel("T (℃)")
            self.ax_fit.set_ylabel("y")
            self.ax_fit.grid('on')
            self.fit_canvas.draw()
        except Exception as e:
            self.ui.statusbar.showMessage('ax_fit plot: '+str(e))
            return

        try:
            T = np.hstack(T)
            y = np.hstack(y)
            T_sort = np.sort(T)
            y_sort = y[np.argsort(T)]
            T_min = self.ui.min_fit_T.value()
            T_max = self.ui.max_fit_T.value()
            if T_min < T_max:
                mask = (T_sort > T_min) & (T_sort < T_max)
                T_sort = T_sort[mask]
                y_sort = y_sort[mask]

            p = np.polyfit(1/T_sort, y_sort, 2)
            y_hat = np.poly1d(p)(1/T_sort)

            X = np.column_stack((1/T_sort**2, 1/T_sort, np.ones(T_sort.shape)))
            res = sm.OLS(y_sort, X).fit()
            pred_ols = res.get_prediction()
            iv_l = pred_ols.summary_frame()["obs_ci_lower"]
            iv_u = pred_ols.summary_frame()["obs_ci_upper"]
            if self.ui.show_pred_interval.isChecked():
                self.ax_fit.fill_between(T_sort, iv_l, iv_u, alpha=0.2)

            formula = r'y={:.2f}/T^2+{:.2f}/T+{:.2f}'.format(p[0], p[1], p[2])
            self.ax_fit.plot(T_sort, y_hat, 'r-', label=formula)
            if self.ui.show_formula.isChecked():
                self.ax_fit.legend()
            self.fit_canvas.draw()
        except Exception as e:
            self.ui.statusbar.showMessage('fit: '+str(e))
            return

        self.ui.btn_export_fit.setEnabled(True)
        self._fit_data = [T, y, T_sort, y_hat, iv_l, iv_u, formula, p]
        self.ui.val_A.setValue(p[0])
        self.ui.val_B.setValue(p[1])
        self.ui.val_C.setValue(p[2])
        self.ui.cbx_analyze_type.setCurrentIndex(
            self.ui.cbx_fit_type.currentIndex())
        self.ui.cbx_analyze_f0.setCurrentIndex(
            self.ui.cbx_fit_base_f.currentIndex())
        self.ui.cbx_analyze_freq_gap.setCurrentIndex(
            self.ui.cbx_fit_freq_gap.currentIndex())

    def export_fit(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(
                None, "Save File", "", "Txt Files (*.txt)")
            if not filename:
                return
            T, y, T_sort, y_hat, iv_l, iv_u, formula, p = self._fit_data
            type_fit = str(self.ui.cbx_fit_type.currentIndex())
            f0 = str(self.ui.cbx_fit_base_f.currentIndex())
            gap = str(self.ui.cbx_fit_freq_gap.currentIndex())
            comments = f'# Y=A/(T^2)+B/(T)+C\n# {p[0]:.4f}\t{p[1]:.4f}\t{p[2]:.4f}\n# {type_fit}\t{f0}\t{gap}\n'
            data_original = np.column_stack([T, y])
            data_fit = np.column_stack([T_sort, y_hat, iv_l, iv_u])
            np.savetxt(filename, data_fit, delimiter='\t', fmt='%.6g',
                       header='# Temperature\ty_hat\ty_pred_low\ty_pred_up', comments=comments)
            if filename.split('.')[-1].lower() == 'txt':
                np.savetxt(filename[:-4]+'_original.txt', data_original,
                           delimiter='\t', header='Temperature\ty')

        except Exception as e:
            self.ui.statusbar.showMessage(str(e))
            return

    # analyze pannel

    def import_fit_param(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", "txt Files (*.txt;);;All Files (*)")
            if not fileName:
                return
            with open(fileName, 'r') as f:
                line = f.readline()
                line = f.readline()
                A, B, C = map(float, line[1:].split('\t'))
                self.ui.val_A.setValue(A)
                self.ui.val_B.setValue(B)
                self.ui.val_C.setValue(C)
                line = f.readline()
                type_fit, f0, gap = map(int, line[2:].split('\t'))
                self.ui.cbx_analyze_type.setCurrentIndex(type_fit)
                self.ui.cbx_analyze_f0.setCurrentIndex(f0)
                self.ui.cbx_analyze_freq_gap.setCurrentIndex(gap)
        except Exception as e:
            self.ui.statusbar.showMessage(
                f'Error when loading parameters: {e}')
            return

    def add_analyze(self):
        fileNames, _ = QFileDialog.getOpenFileNames(
            None, "Select File", "", "CSV Files (*.csv;*.txt;*.pkl;*.xlsx);;All Files (*)")
        if not fileNames:
            return
        for fileName in fileNames:
            row = self.ui.table_analyze.rowCount()
            self.ui.table_analyze.insertRow(row)
            self.ui.table_analyze.setItem(
                row, 0, QTableWidgetItem(fileName))
            self.ui.table_analyze.resizeColumnsToContents()
        self.ui.btn_run_analyze.setEnabled(True)
        self.ui.btn_export_analyze.setEnabled(True)
        if row >= 2:
            self.ui.btn_set_default_ref.setEnabled(True)

    def clear_analyze(self):
        self.ui.table_analyze.setRowCount(0)
        self.ui.btn_run_analyze.setEnabled(False)
        self.ui.btn_export_analyze.setEnabled(False)
        self.ui.btn_set_default_ref.setEnabled(False)

    def delete_analyze(self):
        selected_rows = [index.row()
                         for index in self.ui.table_analyze.selectedIndexes()]
        rows_to_remove = sorted(list(set(selected_rows)), reverse=True)
        for row in rows_to_remove:
            self.ui.table_analyze.removeRow(row)
        row = self.ui.table_analyze.rowCount()
        if row <= 1:
            self.ui.btn_set_default_ref.setEnabled(False)
        if row == 0:
            self.ui.btn_run_analyze.setEnabled(False)
            self.ui.btn_export_analyze.setEnabled(False)

    def run_analyze(self):
        try:
            self._analyze_data = self._run_analyze()
        except Exception as e:
            self.ui.statusbar.showMessage(f'Error when running analysis: {e}')

    def export_analyze(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(
                None, "Save File", "", "Txt Files (*.txt)")
            if not filename:
                return
            df = np.column_stack(self._analyze_data)
            np.savetxt(filename, df, delimiter='\t',
                       fmt='%.6g', header='delta\ty\tT_pred')
        except Exception as e:
            self.ui.statusbar.showMessage(f'Error when saving results: {e}')

    def set_default_ref(self):
        rows = self.ui.table_analyze.rowCount()
        if rows > 0:
            try:
                val_ref = float(self.ui.table_analyze.item(0, 2).text())
                T_ref = float(self.ui.table_analyze.item(0, 3).text())
            except Exception as e:
                self.ui.statusbar.showMessage(f'Error when set default: {e}')
                return
            for i in range(rows):
                self.ui.table_analyze.setItem(
                    i, 2, QTableWidgetItem(f"{val_ref}"))
                self.ui.table_analyze.setItem(
                    i, 3, QTableWidgetItem(f"{T_ref}"))
        self.ui.table_analyze.resizeColumnsToContents()


if __name__ == '__main__':
    app = QApplication([])
    window = EISTMainWindow()
    window.show()
    app.exec_()
