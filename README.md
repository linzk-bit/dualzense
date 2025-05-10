# DualZense - Dual-frequency Impedance Sensing for Temperature Evaluation

A desktop application for electrochemical impedance spectroscopy (EIS) data analysis and temperature evaluation using dual-frequency impedance sensing.

## Features

- **EIS Data Import**: Load EIS data from CSV/TXT/XLSX files or pre-saved PKL datasets.
- **Visualization**: 
  - 3D Nyquist/Bode plots for EIS data preview.
  - Thermal sensitivity maps for different impedance components.
- **Model Fitting**: 
  - Polynomial fitting of temperature sensitivity.
  - Activation energy calculation for Arrhenius analysis.
- **Temperature Evaluation**: 
  - Predict battery temperature using pre-trained models.
  - Batch processing for multiple data files.
- **Data Export**: Save processed data as PKL files or export analysis results as TXT.

## Installation

1. **Prerequisites**:
   - Python 3.7+ 
   - Required packages:
     ```bash
     pip install pyqt5 numpy matplotlib scipy statsmodels pandas pickle5
     ```

2. **Download**:
   ```bash
   git clone https://github.com/linzk-bit/dualzense.git
   cd dualzense
   ```

## Usage

1. **Launch Application**:
   ```bash
   python EIST.py
   ```

2. **Data Import**:
   - Click `Add` to import EIS data files (CSV/TXT/XLSX/PKL).
   - Set data columns: Frequency, Real, Imaginary components.

3. **EIS Preview**:
   - Select plot type from dropdown (Nyquist/Bode).
   - Click `Plot original EIS` to visualize loaded data.

4. **Sensitivity Analysis**:
   - Choose parameters: Frequency gap, display type (Z', Z'', |Z|, Phase).
   - Click `Plot sensitivity map` to generate thermal sensitivity contours.

5. **Model Fitting**:
   - Select base frequency and gap.
   - Click `Fit` to generate polynomial model.
   - Export fitting parameters with `Export Fitting Data`.

6. **Temperature Prediction**:
   - Load EIS data files for prediction.
   - Set reference parameters (ΔRef, TRef).
   - Click `Run` to get temperature predictions.

## File Structure

- `eisdata.py`: Data structure definitions for EIS datasets
  - `BatteryInfo`: Metadata storage
  - `EISBattery`: Impedance measurement data containers
  - `EISDataSet`: Complete dataset management with PKL I/O
- `EIST.py`: Main application logic (PyQt5-based GUI)
- `mainwindow_ui.py`: Auto-generated UI layout (Qt Designer)

## License

MIT License. See `LICENSE` for details.

## Support

For issues/questions, please [open an issue](https://github.com/linzk-bit/dualzense/issues).
