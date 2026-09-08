import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmax, savgol_filter
from sklearn.metrics import root_mean_squared_error
from scipy import interpolate
import sympy
import math

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
np.set_printoptions(legacy='1.25')

# redundant as func alpha does all this
def alpha_mono(pH_val, pKa):
    # for monoprotic species

    H = 10**(-pH_val)
    Ka = 10**(-pKa)

    D = (H + Ka)

    alpha_HA = H / D
    alpha_A = Ka / D

    return alpha_HA, alpha_A

# redundant as func alpha does all this
def alpha_di(pH_val, pKa):
    # for diprotic species - all calculations done on paper using equilibrium constants
    # and the fact that C_T = sum of all species (bar H+)
    # all species are converted to the same form and then alpha = [species] / C_T
    # pKa = [pKa1, pKa2]

    H = 10**(-pH_val)
    Ka_1 = 10**(-pKa[0])
    Ka_2 = 10**(-pKa[1])

    D = (H**2 + H * Ka_1 + Ka_1*Ka_2)
    # D was found constant across all three!
    # D is in fact the sum of the numerators !

    alpha_H2A = (H**2) / D
    alpha_HA = (Ka_1 * H) / D
    alpha_A2 = (Ka_1 * Ka_2) / D

    return alpha_H2A, alpha_HA, alpha_A2


def alpha(pH_val, pKa):
    # this formula is based on the previous 2 formulas as there is a pattern!

    Ka_symbols = []
    for n in range(1, len(pKa)+1): # makes the Ka_1, Ka_2, ...
        k = 'Ka_' + str(n)
        Ka_symbols.append(k)

    symbols = sympy.symbols(Ka_symbols) # symbol[0] = Ka_1, symbol[1] = Ka_2, ...
    H_symbol = sympy.symbols('H')
    # needed to build the expression we need to solve for any n-protic acid

    Ka_list = []
    for vals in pKa:
        Ka_list.append(10**(-vals))

    no_of_protons = len(pKa)
    numerator_H_list = []
    numerator_ka_list = []

    # general order
    # [H^n, H^n-1 k1, H^n-2 k1 k2, H^n-3 k1 k2 k3 , ... , ....]

    count = 0
    expr_list= []

    for protons in range(no_of_protons + 1):
        delta_protons = no_of_protons - protons
        H_expr = H_symbol ** delta_protons

        if count == 0:
            Ka_expr = 1
        if count > 0:
            Ka_expr = sympy.prod(symbols[:count])

        expr = H_expr * Ka_expr
        expr_list.append(expr)
        count += 1

    final_expr = sympy.Add(*expr_list)

    H_val = 10 ** (-pH_val)

    denominator = []
    alpha_list = []  # one entry per pH value, each entry is a list of alphas for that pH

    for pH in H_val:

        Ka_subs = dict(zip(symbols[0:], Ka_list))
        subs_dict = {H_symbol: pH}
        subs_dict.update(Ka_subs)

        val = final_expr.evalf(subs=subs_dict)
        D = val # this is the denominator value ---
        denominator.append(D)

        alphas_this_pH = []
        for expression in expr_list:
            res = expression.evalf(subs=subs_dict)
            alphas_this_pH.append(res / D)

        alpha_list.append(alphas_this_pH)
        alphas_by_species = list(zip(*alpha_list))

        alphas_by_species = np.array(alphas_by_species)

    return alphas_by_species


class Titration:
    def __init__(self, df):
        self.df = pd.read_csv(df)
        self.x_vals = self.df.iloc[:,0]
        self.y_vals = self.df.iloc[:,1]
        self.x_col = self.df.columns[0]
        self.y_col = self.df.columns[1]

    def smooth(self, method, order=None):

        rmse_list = []
        self.method = method

        # The order refers both to the n_poly order in Savgol and also the 'resolution' in Spline!

        if method == 'Savgol':
            # perform a quick optimisation:
            for n in range(order+1, order+6):
                y_smooth = savgol_filter(self.y_vals, n, order, mode = 'nearest')
                rmse = root_mean_squared_error(self.y_vals, y_smooth)
                rmse_list.append(rmse)

            optimal_n_index = rmse_list.index(min(rmse_list))
            optimal_n = optimal_n_index + order+1

            y_smooth = savgol_filter(self.y_vals, optimal_n, order, mode='nearest')

            plt.plot(self.x_vals, self.y_vals, label='Actual Data')
            plt.plot(self.x_vals, y_smooth, label='Smoothed', ls='--')
            plt.xlabel(self.x_col)
            plt.ylabel(self.y_col)
            plt.title(f'Smoothened Titration Data - Window = {optimal_n}')
            plt.legend()
            plt.show()

            self.smooth = True
            self.y_vals = y_smooth

        if method == 'Spline':

            for n in range(0,5):
                y_smoothener = interpolate.make_smoothing_spline(self.x_vals, self.y_vals, lam=n)
                rmse = root_mean_squared_error(self.y_vals, y_smoothener(self.x_vals))
                rmse_list.append(rmse)

            optimal_lam = rmse_list.index(min(rmse_list)) + 1

            y_smoothener = interpolate.make_smoothing_spline(self.x_vals, self.y_vals, lam=optimal_lam)

            smooth_x = np.linspace(min(self.x_vals), max(self.x_vals), int(len(self.x_vals)*order))
            smooth_y = y_smoothener(smooth_x)

            plt.plot(self.x_vals, self.y_vals, label='Actual Data')
            plt.plot(smooth_x, smooth_y, label='Smoothed', ls='--')
            plt.xlabel(self.x_col)
            plt.ylabel(self.y_col)
            plt.title(f'Smoothened Titration Data - Lam = {optimal_lam}')
            plt.legend()
            plt.show()

            self.smooth = True
            self.y_vals = smooth_y
            self.x_vals = smooth_x

            self.df = pd.DataFrame({self.x_col:self.x_vals, self.y_col:self.y_vals})

    def plot(self):

        fig, axs = plt.subplots(2,2, figsize=(12,8))

        if hasattr(self, 'smooth'):
            plt.suptitle(f'Smoothened Titration - {self.method}')
            axs[0, 0].plot(self.x_vals, self.y_vals)
        else:
            plt.suptitle('Titration')
            axs[0, 0].plot(self.x_vals, self.y_vals, marker='o', markersize=5)

        axs[0, 0].set_title('Titration Curve')
        axs[0, 0].set_xlabel(self.df.columns[0])
        axs[0, 0].set_ylabel(self.df.columns[1])

        # pka = to the pH equivalence point (sharp slope)
        # first derivative (dy/dx = y, x = x)
        # second derivative (dydy/dx = y, x = x)

        # to determine pKa -> find half equivalence point ... half the V at equivalence point!
        # should look at +- V/2 to find upper pKas if there are any..

        self.dy = np.gradient(self.y_vals, self.x_vals)
        self.dx = np.gradient(self.x_vals, self.x_vals)

        self.dydy = np.gradient(self.dy, self.x_vals)

        self.dydx = self.dy / self.dx
        self.dydydx = self.dydy / self.dx

        min_prominence = 0.25 * max(self.dydydx)

        peak, props = find_peaks(self.dydx, plateau_size=1, width=2, prominence= min_prominence)
        print('Detected equivalence points:')
        print(self.df.iloc[peak].round(2).to_string(index=False))

        print('Half-equivalence volumes:')

        for n in self.df.iloc[peak].index:
            point = self.df.iloc[peak][self.x_col][n]
            half_point = point/2

            low, high = [half_point, half_point + point]
            print(round(low,2), round(high,2))


        axs[0,1].plot(self.x_vals, self.dydx)
        axs[0,1].scatter(self.df.iloc[peak][self.x_col].values, self.df.iloc[peak][self.y_col].values, color='r', marker='x',label='Local Max')

        for n in range(len(peak)):
            axs[0,1].axvline(self.df.iloc[peak][self.x_col].values[n], color='r', ls='--',alpha=0.5)

        axs[0,1].set_title('First Derivative - Equivalence Points')
        axs[0,1].set_ylim(0, max(self.dydx) * 1.2)
        axs[0,1].set_ylabel(f'd{self.df.columns[1]}/dV')
        axs[0,1].set_xlabel(self.df.columns[0])

        axs[1, 0].plot(self.x_vals,self.dydydx)

        for n in range(len(peak)):
            axs[1, 0].axvline(self.df.iloc[peak][self.x_col].values[n], color='r', ls='--', alpha=0.5)

        axs[1, 0].set_title('Second Derivative - Equivalence Points')
        axs[1, 0].set_ylabel(f'd²{self.df.columns[1]}/dV²')
        axs[1, 0].set_xlabel(self.df.columns[0])

        axs[1, 1].plot(self.x_vals, self.y_vals)
        axs[1, 1].set_title('Equivalence Points')
        axs[1, 1].set_xlabel(self.df.columns[0])
        axs[1, 1].set_ylabel(self.df.columns[1])

        for n in range(len(peak)):
            axs[1, 1].axvline(self.df.iloc[peak][self.x_col].values[n], color='r', ls='--',alpha=0.5)

        plt.tight_layout()
        plt.show()

    def species_plot(self, pKa, name=None):
        # get the amount of each species at certain pH (estimate)
        # for monoprotic, alpha should equal [H+] / ka
        # [H+] = 10^-pH, ka is 10^-pKa from plot - need to get manually atm..
        # shortened: a = 10^(-pH + pka)

        self.pKa = pKa

        pH_vals = np.linspace(0,14,100)
        plot = alpha(pH_vals, self.pKa)

        colours = ['#6F87AD' ,'#8391C1' ,'#979BD5' ,'#ABA5E9' ,'#BFAFFD' ,'#D3B9FF' ,'#E7C3FF' ,'#FBCDFF' ,'#FFD7FF' ,'#FFE1FF']
        n_ic = ['Mono', 'Di', 'Tri']

        for n in range(plot.shape[0]):
            if name is not None:
                plt.plot(pH_vals, plot[n], label=name[n])
            else:
                plt.plot(pH_vals, plot[n])

        for i in range(len(self.pKa)):
            plt.axvline(self.pKa[i], color='r', ls='--', alpha=0.5)

        if len(self.pKa) <= 3:
            n_title = n_ic[len(self.pKa) - 1] # -1 as this includes the non-deprotonated part
        else:
            n_title = 'Poly'

        plt.title(f'{n_title}-Protic Species Plot')
        plt.legend()
        plt.xlabel(self.y_col)


        plt.ylabel('α (Fraction of species)')
        plt.show()


t = Titration('titration_triprotic_noisy.csv')
t.smooth(method = 'Spline', order = 2)
t.plot()
t.species_plot(pKa = [2.34, 9.60], name = ['Gly (+1)','Gly (0)','Gly (-1)'])

