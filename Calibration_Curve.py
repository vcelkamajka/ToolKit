import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

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
        self.run = self.dataframe.columns[0]
        self.x_col = self.dataframe.columns[1]
        self.y_col = self.dataframe.columns[2]

        self.x_vals = self.dataframe[self.x_col]
        self.y_vals = self.dataframe[self.y_col]

        self.dataframe.set_index(self.run, inplace=True)

        print(self.dataframe)

        no_of_runs = self.dataframe.index[-1]
        self.run_list = []

        for n in range(no_of_runs + 1):
            new_var = make_placeholder(n)
            self.run_list.append(new_var)

        count = 0
        y_std_list = []
        y_mean_list = []
        x_std_list = []
        x_mean_list = []
        for run in self.run_list:
            run = self.dataframe[self.dataframe[self.x_col] == self.x_vals[count]]
            run_x_only = run[self.x_col]
            run_y_only = run[self.y_col]
            count += 1

            x_std_list.append(get_std(run_x_only))
            x_mean_list.append(get_mean(run_x_only))

            y_std_list.append(get_std(run_y_only))
            y_mean_list.append(get_mean(run_y_only))

        stats_df = pd.DataFrame()
        stats_df['Run No.'] = self.run_list
        stats_df[f'{self.y_col} stdev (sample)'] = y_std_list
        stats_df[f'{self.y_col} mean'] = y_mean_list
        stats_df[f'{self.x_col} stdev (sample)'] = x_std_list
        stats_df[f'{self.x_col} mean'] = x_mean_list
        stats_df = stats_df.reset_index(drop=True)
        print(stats_df)

        stats_df.to_csv('Calibration_Stats.csv', index=False)
        self.stats_df = stats_df


    def get_curve(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        x = self.stats_df[f'{self.x_col} mean'].to_numpy()
        x_2d = x.reshape(-1, 1)

        x_error = self.stats_df[f'{self.x_col} stdev (sample)'].to_numpy()

        y = self.stats_df[f'{self.y_col} mean'].to_numpy()
        y_2d = y.reshape(-1, 1)

        y_error = self.stats_df[f'{self.y_col} stdev (sample)'].to_numpy()

        model = LinearRegression()
        model.fit(x_2d, y_2d)

        m = model.coef_[0]
        c = model.intercept_
        y_pred = model.predict(x_2d)
        r2 = r2_score(y_2d, y_pred)

        ax.errorbar(x,y, xerr= x_error, yerr= y_error, marker='o',solid_capstyle='projecting', capsize=5,label='Data')

        if c.item() < 0.001:
            ax.plot(x,y_pred, color='r', ls='--',label=f'y = {m.item():.3f}x + {c.item():.3e}\nR² = {r2:.3f}')
        else:
            ax.plot(x, y_pred, color='r', ls='--', label=f'y = {m.item():.3f}x + {c.item():.3f}\nR² = {r2:.3f}')

        ax.set_xlabel(self.x_col)
        ax.set_ylabel(self.y_col)
        ax.grid(alpha=0.5, linestyle=':')
        ax.legend()
        plt.show()


t = Calibration('calibration_data.csv')
t.get_curve()

