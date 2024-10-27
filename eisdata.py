from dataclasses import dataclass
import numpy as np
import pickle


@dataclass
class BatteryInfo:
    cycle_temperature: list[int] | None = None
    cycle_crate: list[float] | None = None
    cycle_number: list[float] | None = None
    cell_type: list[str] | None = ''
    cell_serial: list[str] | None = ''

    def extend(self, other: "BatteryInfo"):
        if not isinstance(other, BatteryInfo):
            raise TypeError("Can only append BatteryInfos")
        if self.cycle_temperature is None:
            self.cycle_temperature = []
            self.cycle_crate = []
            self.cycle_number = []
            self.cell_type = []
            self.cell_serial = []
        if not isinstance(other.cycle_temperature, list):
            other.cycle_temperature = [other.cycle_temperature]
        if not isinstance(other.cycle_crate, list):
            other.cycle_crate = [other.cycle_crate]
        if not isinstance(other.cycle_number, list):
            other.cycle_number = [other.cycle_number]
        if not isinstance(other.cell_type, list):
            other.cell_type = [other.cell_type]
        if not isinstance(other.cell_serial, list):
            other.cell_serial = [other.cell_serial]

        self.cycle_temperature.extend(other.cycle_temperature)
        self.cycle_crate.extend(other.cycle_crate)
        self.cycle_number.extend(other.cycle_number)
        self.cell_type.extend(other.cell_type)
        self.cell_serial.extend(other.cell_serial)


@dataclass
class EISBattery:
    temperature: list[list[float]]
    frequency: list[list[np.ndarray]]
    real: list[list[np.ndarray]]
    neg_imag: list[list[np.ndarray]]

    def extend(self, other: "EISBattery"):
        if not isinstance(other, EISBattery):
            raise TypeError("Can only append EISBatteries")
        if self.temperature is None:
            self.temperature = []
            self.frequency = []
            self.real = []
            self.neg_imag = []
        if not isinstance(other.temperature, list):
            other.temperature = [other.temperature]
        if not isinstance(other.frequency, list):
            other.frequency = [other.frequency]
        if not isinstance(other.real, list):
            other.real = [other.real]
        if not isinstance(other.neg_imag, list):
            other.neg_imag = [other.neg_imag]
        self.temperature.extend(other.temperature)
        self.frequency.extend(other.frequency)
        self.real.extend(other.real)
        self.neg_imag.extend(other.neg_imag)


@dataclass
class EISDataSet:
    soc: list[float] = None
    soh: list[float] = None
    info: BatteryInfo = None
    eis: EISBattery = None

    def __len__(self) -> int:
        return len(self.soc)

    def get_eis(self, index=0) -> EISBattery:
        return EISBattery(
            list(self.eis.temperature[index]),
            list(self.eis.frequency[index]),
            list(self.eis.real[index]),
            list(self.eis.neg_imag[index]))

    def extend(self, other: "EISDataSet"):
        if not isinstance(other, EISDataSet):
            raise TypeError("Can only append EISDataSets")
        if self.soc is None:
            self.soc = []
            self.soh = []
        if not isinstance(other.soc, list):
            self.soc.extend([other.soc])
            self.soh.extend([other.soh])
        else:
            self.soc.extend(other.soc)
            self.soh.extend(other.soh)
        self.info.extend(other.info)
        self.eis.extend(other.eis)

    def to_pkl(self, fileName) -> None:
        data = {
            'soc': self.soc,
            'soh': self.soh,
            'info': {
                'cycle_temperature/C': self.info.cycle_temperature,
                'cycle_crate': self.info.cycle_crate,
                'cycle_number': self.info.cycle_number,
                'cell_type': self.info.cell_type,
                'cell_serial': self.info.cell_serial,
            },
            'eis': {
                'temperature/C': self.eis.temperature,
                'frequency/Hz': self.eis.frequency,
                'real/ohm': self.eis.real,
                '-imag/ohm': self.eis.neg_imag,
            }
        }
        with open(fileName, 'wb') as f:
            pickle.dump(data, f)

    def from_pkl(fileName):
        with open(fileName, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, EISDataSet):
            return data
        elif isinstance(data, dict):
            info = BatteryInfo(data['info']['cycle_temperature/C'],
                            data['info']['cycle_crate'],
                            data['info']['cycle_number'],
                            data['info']['cell_type'],
                            data['info']['cell_serial'])
            eis = EISBattery(data['eis']['temperature/C'],
                            data['eis']['frequency/Hz'],
                            data['eis']['real/ohm'],
                            data['eis']['-imag/ohm'])
            dataset = EISDataSet(data['soc'], data['soh'], info, eis)
            return dataset


if __name__ == '__main__':
    fileName = './testdata/1C-2-532.pkl'
    dataset = EISDataSet.from_pkl(fileName)
    for i in range(len(dataset)):
        print(dataset.get_eis(i).temperature)
