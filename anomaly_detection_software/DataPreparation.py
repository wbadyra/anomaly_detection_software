import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data, round_half_away_from_zero
from faults_preparation import insert_fault_erratic, insert_fault_hardover, insert_fault_spike, insert_fault_drift, \
    insert_empty_error_columns
from preprocessing import filter_by_zeros, filter_out_negatives
import seaborn as sns


def visualize_data_time(data, params, start_index=0, period_length=10000):
    """
                visualize_data_time(data, params, start_index=0, period_length=10000):
                    Plot of selected parameters of data DataFrame, in time, for selected time range
    """
    for i in range(len(params)):
        plt.subplot(len(params), 1, i + 1)
        plt.plot(data[start_index:start_index + period_length].index,
                 data[start_index:start_index + period_length][params[i]], color='red')
        plt.ylabel(params[i])

    plt.xlabel('index')
    plt.show()


class DataPreparation:
    """Class containing multiple dataframes with differently processed data

            Attributes
            ----------
            data : DataFrame
                data to work with
            data_without_errors : DataFrame
                data straight form the car -  without inserting errors
            data_drive : DataFrame
                subset of data with velocity greater than 0
            data_stationary : DataFrame
                subset of data, where velocity is close to 0
            scaler : sklearn.preprocessing.StandardScaler
                used for normalizing data DataFrame
            data_scaled : DataFrame
                data normalized to 0-1 range
            data_drive_scaled : DataFrame
                data_drive normalized to 0-1 range
            data_stationary_scaled : DataFrame
                data_stationary normalized to 0-1 range


            Methods
            -------
            visualize_data():
                Shows scatter plot of column1 vs column2, where columns can be selected from data
            visualize_data_error_comparison(params, start_index=0, period_length=10000):
                Plot of selected parameters in time, comparing plain data to data with inserted errors
            visualize_data_time(data, params, start_index=0, period_length=10000):
                Plot of selected parameters of data DataFrame, in time, for selected time range
            data_correlation():
                Correlation between each 2 data columns

    """

    def __init__(self, filename=None):
        # data reading and scaling
        self.data = read_data(filename=filename)
        self.cols = list(self.data.columns)
        self.cols.remove('Time')
        # drop for driving data kia soul
        # self.data = self.data.drop(['Time(s)', 'Class', 'PathOrder'], axis=1)
        self.data = self.data.drop(['Time', 'index'], axis=1, errors='ignore')
        
        self.data_correlation()
        
        self.data = filter_out_negatives(self.data, ['Vehicle Speed [km/h]', 'Accelerator Pedal Position [%]'])
        
        print("Done that")
        
        self.data_without_errors = self.data.copy(deep=True)
        self.data = insert_empty_error_columns(self.data)

    def insert_error(self, type='spike', errors_col=['Engine RPM [RPM]', 'Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]']):
        # instering different fault types
        #if type == 'erratic':
        self.data_drive_err = insert_fault_erratic(self.data_drive.copy(deep=True), errors_col,
                                                  intensity=1, time=0.05, period_length=15, same_periods=False)
        #elif type == 'hardover':
        self.data_drive_err2 = insert_fault_hardover(self.data_drive.copy(deep=True), errors_col, time=0.05, period_length=15, same_periods=False)
        #elif type == 'spike':
        self.data_drive_err3 = insert_fault_spike(self.data_drive.copy(deep=True), errors_col,
                                              intensity = 2, time=0.05, same_periods=False)
        #elif type == 'drift':
        self.data_drive_err4 = insert_fault_drift(self.data_drive.copy(deep=True), errors_col,
                                          intensity=1.5, time=0.05, period_length=20, same_periods=False)

    def filter_out_statinary_and_drive_data(self):
        self.data_drive, self.data_stationary = \
            filter_by_zeros(self.data, ['Vehicle Speed [km/h]'])
        if 'Accelerator Pedal Position [%]' in self.data.columns:
            self.data_drive, self.data_drive_acc0 = \
                filter_by_zeros(self.data_drive, ['Accelerator Pedal Position [%]'])

    def scale_all_data(self):
        # data scaling
        self.scaler = MinMaxScaler().fit(self.data[self.cols])
        self.data_scaled = self.data.copy(deep=True)
        self.data_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data[self.cols]), columns=self.cols)
        self.data_drive_scaled = self.data_drive.copy(deep=True)
        self.data_drive_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data_drive[self.cols]),
                                                         columns=self.cols)
        self.data_stationary_scaled = self.data_stationary.copy(deep=True)
        self.data_stationary_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data_stationary[self.cols]),
                                                              columns=self.cols)
        self.data_drive_scaled_err = self.data_drive_err.copy(deep=True)
        self.data_drive_scaled_err[self.cols] = pd.DataFrame(self.scaler.transform(self.data_drive_err[self.cols]),
                                                             columns=self.cols)


    def visualize_data(self):
        # Data visualization
        plt.figure(figsize=(8, 6))
        # plt.scatter(self.data.Fuel_consumption, self.data.Accelerator_Pedal_value, s=0.2)
        # plt.scatter(self.data_scaled.Fuel_consumption, self.data_scaled.Intake_air_pressure, s=0.2)
        plt.scatter(self.data.Fuel_consumption, self.data.Intake_air_pressure, s=0.2)
        plt.xlabel('fuel_cons')
        # plt.ylabel('pedal val')
        plt.show()

    def visualize_data_error_comparison(self, params, start_index=0, period_length=10000):
        # Data visualization

        # for i in range(4):
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax1.plot(self.data_drive_err[start_index:start_index + period_length].index,
                    self.data_drive_err[start_index:start_index + period_length][params[0]], label='zakłócenia', color='red')
        ax1.plot(self.data_drive.iloc[start_index:start_index + period_length].index,
                    self.data_drive[start_index:start_index + period_length][params[0]], label='sygnał poprawny',color='blue')
        ax1.set(ylabel="RMP")     
        ax1.title.set_text("erratic")
        
        ax2 = fig.add_subplot(412)
        ax2.plot(self.data_drive_err2[start_index:start_index + period_length].index,
                    self.data_drive_err2[start_index:start_index + period_length][params[0]], color='red')
        ax2.plot(self.data_drive.iloc[start_index:start_index + period_length].index,
                    self.data_drive[start_index:start_index + period_length][params[0]], color='blue')
        ax2.set(ylabel="RMP")     
        ax2.title.set_text("hardover")

        
        ax3 = fig.add_subplot(413)
        ax3.plot(self.data_drive_err3[start_index:start_index + period_length].index,
                    self.data_drive_err3[start_index:start_index + period_length][params[0]], color='red')
        ax3.plot(self.data_drive.iloc[start_index:start_index + period_length].index,
                    self.data_drive[start_index:start_index + period_length][params[0]], color='blue')
        ax3.set(ylabel="RMP")     
        ax3.title.set_text("spike")

        ax4 = fig.add_subplot(414)
        ax4.plot(self.data_drive_err4[start_index:start_index + period_length].index,
                    self.data_drive_err4[start_index:start_index + period_length][params[0]], color='red')
        ax4.plot(self.data_drive.iloc[start_index:start_index + period_length].index,
                    self.data_drive[start_index:start_index + period_length][params[0]], color='blue')
        ax4.set(ylabel="RMP")     
        ax4.title.set_text("drift")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def data_correlation(self):
        corr = self.data[self.cols].corr(method='pearson')
        print(corr)
        names = list(corr.columns)
        names = [n[0: n.find('[') - 1] for n in names]
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.matshow(corr, cmap='coolwarm')
        ax.set_xticks(np.arange(0, 9, 1))
        ax.set_yticks(np.arange(0, 9, 1))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.xticks(rotation=45, ha='left')
        for (i, j), z in np.ndenumerate(corr):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.tight_layout()
        plt.show()