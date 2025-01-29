import peakutils
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed, compute
from scipy import interpolate, optimize
from scipy.stats import wasserstein_distance
from numpy import trapz
from file_reader_sorter_parser import SortList
import glob





class Processor(SortList):

    def __init__(self, nv_type='nv', nv_zpl=[634.25, 640.25], nv0_zpl=[572.0, 578],
                 huang_rhys=[649, 780], k=['Frame-00001', 'Frame-0001']):
        super().__init__()
        self.nv_zpl = nv_zpl
        self.nv0_zpl = nv0_zpl
        self.huang_rhys = huang_rhys
        self.nv_type = nv_type
        self.k = ['Frame-00001', 'Frame-0001']
        self.filtered_files = []

    
    @staticmethod
    def filter_list(self):
        filenames 
        if len(self.k) < 2:
            self.filtered_files = [name for name in filenames if self.k not in name]
        else:
            self.filtered_files = [name for name in filenames if all(k not in name for k in self.k)]
        return self.filtered_files

    @staticmethod
    def gaussian(x_zpl, amp, u, std):
        return amp * np.exp(-((x_zpl - u) ** 2 / (2 * std ** 2)))

    @staticmethod
    def lorentzian(x_zpl, x0, a, gam):
        return a * gam ** 2 / (gam ** 2 + (x_zpl - x0) ** 2)
    
    def spline_fit(self, x_zpl, y_zpl_base):
        tck_zpl = interpolate.splrep(x_zpl, y_zpl_base, s=0.0001)
        x_zpl_sim = np.arange(np.min(x_zpl), np.max(x_zpl), 0.1)
        y_zpl_sim = interpolate.splev(x_zpl_sim, tck_zpl, der=0)
        HM = (np.max(y_zpl_sim) - np.min(y_zpl_sim)) / 2
        w = interpolate.splrep(x_zpl_sim, y_zpl_sim - HM, k=3)
        roots = interpolate.sproot(w)
        if len(roots) == 2:
            r1, r2 = roots
            FWHM = np.abs(r1 - r2)
            center_wavelength = r1 + FWHM / 2
            return center_wavelength, FWHM
        return None, None

    def process_file(self, f1):
        if self.nv_type == 'nv':
            zp = self.nv_zpl
        else:
            zp = self.nv0_zpl

        result = {'frame_num': self.file_num(f1)}

        # Read CSV with Dask
        df = dd.read_csv(f1, sep=',', header=0)
        df = df.sort_values(by='Wavelength').drop_duplicates(subset='Wavelength', keep='first').compute()

        x, y = df['Wavelength'].values, df['Intensity'].values
        x_zpl = x[(np.abs(x - zp[0])).argmin():(np.abs(x - zp[1])).argmin()]
        y_zpl = y[(np.abs(x - zp[0])).argmin():(np.abs(x - zp[1])).argmin()]

        # Remove baseline
        base = peakutils.baseline(y_zpl, 1)
        y_zpl_base = y_zpl - base

        # Compute area statistics
        dx_val = (x[0] - x[50]) / 50
        area_zpl = trapz(y[(np.abs(x - zp[0])).argmin():(np.abs(x - zp[1])).argmin()], dx=dx_val)
        area_psb = trapz(y[(np.abs(x - self.huang_rhys[0])).argmin():(np.abs(x - self.huang_rhys[1])).argmin()], dx=dx_val)
        dw = area_zpl / area_psb
        result.update({'time': self.time_st(f1), 'debye_waller': dw})

        # Fit the ZPL
        popt, pcov = None, None
        if self.fit_type == 'gaussian':
            popt, pcov = optimize.curve_fit(self.gaussian, x_zpl, y_zpl_base, [4000, 637.5, 1.5], maxfev=50000)
        elif self.fit_type == 'lorentzian':
            popt, pcov = optimize.curve_fit(self.lorentzian, x_zpl, y_zpl_base, [4000, 637.5, 1.5], maxfev=50000)
        elif self.fit_type == 'spline':
            center_wavelength, FWHM = self.spline_fit(x_zpl, y_zpl_base)
            result.update({'peak_center': center_wavelength, 'width': FWHM})
            return result

        if popt is not None:
            amp, center_wavelength, FWHM = popt
            result.update({'peak_center': center_wavelength, 'width': FWHM, 'amplitude': amp,
                           'laser_pow': self.laser_power(f1), 'temperature': float(self.temp(f1))})

        return result

    def create_dataframe(self, results):
        return pd.DataFrame(results)

    def export_dataframe(self, df, export_name='name_me'):
        df.to_csv(f'../saved_data/{export_name}', index=False)

def main():
    client = Client()  # Start a Dask client

    processor = Processor()
    processor.filter_list(SortList)

    # Process files in parallel using Dask
    results = dask.compute(*(delayed(processor.process_file)(f) for f in processor.filtered_files))
    df = processor.create_dataframe(results)

    # Export the final DataFrame
    processor.export_dataframe(df, export_name='processed_data')

if __name__ == '__main__':
    main()

