import pandas as pd
import numpy as np
import matplotlib as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# need to a) read csv file of data, b) get the repeats in order to find error, c) plot results + error and d) get r2 + error

class Calibration:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.dataframe = pd.read_csv(dataframe)
        self.run = self.dataframe.columns[0]
        self.x_col = self.dataframe.columns[1]
        self.y_col = self.dataframe.columns[2]

        self.dataframe.set_index(self.run, inplace=True)
        print(self.dataframe)

        no_of_runs = self.dataframe.index[-1]

        for n in range(no_of_runs):
            




    def get_curve(self):
        pass



t = Calibration('calibration_data.csv')



