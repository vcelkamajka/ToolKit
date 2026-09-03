import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
np.set_printoptions(legacy='1.25')

def make_placeholder(n):
    var = 'Run ' + str(n+1)
    return var

def get_std(x_run):
    std = np.std(x_run, ddof=1)
    return std

def get_mean(x_run):
    mean = np.mean(x_run)
    return mean

# need to a) read csv file of data, b) get the repeats in order to find error, c) plot results + error and d) get r2 + error bars

class Calibration:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.dataframe = pd.read_csv(dataframe)

        # where Run == 1 and Conc. == val -> get mean abs. and stdev

        self.x_col = self.dataframe.columns[1]
        self.y_col = self.dataframe.columns[2]

        conc_range_list = []
        conc_range_list.append(self.dataframe[self.x_col].unique())
        # gets all the UNIQUE conc. values

        res_list = []
        # list in the format of : [ [mean_1, std_1], [mean_2, std_2], ... [...] ]

        for n in range(len(conc_range_list[0])):
            res = self.dataframe[self.dataframe[self.x_col] == conc_range_list[0][n]]
            res = res[self.y_col]

            res_mean = get_mean(res)
            res_std = get_std(res)

            res_list.append([res_mean, res_std])

        len_runs = len(res_list)

        mean_col = f'{self.y_col} Mean'
        std_col = f'{self.y_col} Stdev (Sample)'

        res_df = pd.DataFrame(res_list, columns=[mean_col, std_col])

        res_df.set_index(np.arange(1,len_runs+1,1), inplace=True)
        res_df = res_df.rename_axis('Replicate')

        print(res_df)
        res_df.to_csv('Calibration_Stats.csv')

        fig, ax = plt.subplots()

        x = conc_range_list[0]
        y = res_df[mean_col]
        yerror = res_df[std_col]

        model = LinearRegression()
        model.fit(np.array(x).reshape(-1,1),y)
        preds = model.predict(np.array(x).reshape(-1,1))

        m = model.coef_[0]
        c = model.intercept_
        r2 = r2_score(y,preds)

        ax.errorbar(x,y,xerr=0,yerr=yerror, marker='o', label='Data',solid_capstyle='projecting', capsize=5,ls='')
        if c.item() < 0.001:
            ax.plot(x, preds, color='r', ls='--', label=f'y = {m.item():.3f}x + {c.item():.3e}\nR² = {r2:.3f}')
        else:
            ax.plot(x, preds, color='r', ls='--', label=f'y = {m.item():.3f}x + {c.item():.3f}\nR² = {r2:.3f}')

        ax.set_xlabel(self.x_col)
        ax.set_ylabel(self.y_col)
        ax.grid(alpha=0.5, linestyle=':')
        ax.legend()

        plt.show()


t = Calibration('calibration_data.csv')
t = Calibration('replicate_data.csv')
