import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

# to determine the order of a reaction

def AIC(n,k, y_pred, y_data):
    sse = np.sum((y_pred - y_data)**2)
    return n * np.log(sse/n) + 2*k


def Korsmeyer_Peppas(t, k,n):
    # n ≤ 0.45 to 0.5: Fickian diffusion (the drug simply moves out through pores or the matrix by normal molecular diffusion)
    # 0.45 < n < 0.89 (or 0.5 < n < 1.0): Anomalous (non-Fickian) transport (both diffusion and the relaxing/swelling of the polymer control the release).
    # n = 0.89 to 1.0 (Case-II transport): Erosion or relaxation-controlled release (polymer swelling or chain relaxation drives the release).
    return k*t**n

def Higuchi(t,k):
    return k * np.sqrt(t)

def Hixson_Crowell(t,k,init_amount):
    return -k * t + (init_amount) ** 1/3

def Gompertz(t,max,a,k):
    return max * np.exp(-a*np.exp(-t*k))

def zero_order(t,k):
    return k * t

def first_order(t,k):
    return 100 * (1- np.exp(k * t))


function_list = [zero_order, first_order, Korsmeyer_Peppas, Higuchi, Hixson_Crowell,Gompertz]
function_names = ['Zero Order Model', 'First Order Model', 'Korsmeyer-Peppas Model', 'Higuchi Model', 'Hixson-Crowell Model','Gompertz Model']

class ReleaseModel:
    def __init__(self,filename,df=None,x_vals=None,y_vals=None):
        self.df = pd.read_csv(filename)
        self.x_vals = self.df.iloc[:,0]
        self.y_vals = self.df.iloc[:,1]

        print(self.df)
        print('='*50)

        results_df = pd.DataFrame()
        r2_list = []
        MAPE_list = []
        aic_list = []

        count = 0
        for functions in function_list:
            function = functions
            print(f'MODEL = {function_names[count]}')

            param, param_cov = curve_fit(function, self.x_vals, self.y_vals)
            fitted_y = function(self.x_vals, *param)

            aic = AIC(len(self.x_vals), len(param) + 1, fitted_y, self.y_vals)
            r2 = r2_score(self.y_vals, fitted_y)
            mse = mean_squared_error(fitted_y, self.y_vals)
            mape = mean_absolute_percentage_error(self.y_vals, fitted_y)

            print(f'R² score: {r2:.2f}')
            print(f'MSE score: {mse:.2f}')
            print(f'MAPE score: {mape:.2f}%')
            print(f'AIC score: {aic:.2f}')
            print('=' * 50)

            r2_list.append(r2)
            MAPE_list.append(mape)
            aic_list.append(aic)

            print(f'Condition number of covariance matrix: {np.linalg.cond(param_cov):.2f}')
            print('High condition numbers indicate over parametrising, reduce the number terms.')

            for paramater in param:
                print(f'Paramater: {paramater:.3f}')
            print(param_cov)
            print('='*50)


            plt.scatter(self.x_vals,self.y_vals,label=f'Actual Data')
            if function == Korsmeyer_Peppas:
                plt.plot(self.x_vals,fitted_y,label=f'Fitted Data: R² = {r2:.2f}\nn = {param[1]:.2f}',marker='x',color='#FF7A04')
            else:
                plt.plot(self.x_vals, fitted_y, label=f'Fitted Data: R² = {r2:.2f}', marker='x',color='#FF7A04')
            plt.legend()
            plt.xlabel(self.df.columns[0])
            plt.ylabel(self.df.columns[1])
            plt.title(function_names[count])
            plt.show()
            count += 1

        results_df['Model'] = function_names
        results_df['R²'] = r2_list
        results_df['MAPE'] = MAPE_list
        results_df['AIC'] = aic_list

        results_df.sort_values(by=['R²'],ascending=False,inplace=True)

        results_df = results_df.round(2)
        print(results_df)
        results_df.to_csv('Diffusion_Model_Results.csv')


test = ReleaseModel('drug_release.csv')



def Langmuir(c,qmax,constant):
    return (qmax * constant * c) / (1 + constant*c)

def Freundlich(c,m,k,n):
    return m * k * c ** (1/n)

surface_function_list = [Langmuir,Freundlich]
surface_function_names = ['Langmuir Model', 'Freundlich Model']


class SurfaceModel:
    def __init__(self,filename,df=None,x_vals=None,y_vals=None):
        self.df = pd.read_csv(filename)
        self.x_vals = self.df.iloc[:,0]
        self.y_vals = self.df.iloc[:,1]

        print(self.df)
        print('=' * 50)

        results_df = pd.DataFrame()
        r2_list = []
        MAPE_list = []
        aic_list = []

        count = 0
        for functions in surface_function_list:
            function = functions
            print(f'MODEL = {surface_function_names[count]}')

            if function == Langmuir:
                param, param_cov = curve_fit(function, self.x_vals, self.y_vals,bounds=([0,0],[np.inf,np.inf]))
            else:
                param, param_cov = curve_fit(function, self.x_vals, self.y_vals)


            fitted_y = function(self.x_vals, *param)

            aic = AIC(len(self.x_vals), len(param) + 1, fitted_y, self.y_vals)
            r2 = r2_score(self.y_vals, fitted_y)
            mse = mean_squared_error(fitted_y, self.y_vals)
            mape = mean_absolute_percentage_error(self.y_vals, fitted_y)

            print(f'R² score: {r2:.2f}')
            print(f'MSE score: {mse:.2f}')
            print(f'MAPE score: {mape:.2f}%')
            print(f'AIC score: {aic:.2f}')
            print('=' * 50)

            r2_list.append(r2)
            MAPE_list.append(mape)
            aic_list.append(aic)

            print(f'Condition number of covariance matrix: {np.linalg.cond(param_cov):.2f}')
            print('High condition numbers indicate over parametrising, reduce the number terms.')

            for paramater in param:
                print(f'Paramater: {paramater:.3f}')
            print(param_cov)
            print('=' * 50)

            plt.scatter(self.x_vals, self.y_vals, label=f'Actual Data')
            plt.plot(self.x_vals, fitted_y, label=f'Fitted Data: R² = {r2:.2f}', marker='x',color='#FF7A04')
            plt.legend()
            plt.xlabel(self.df.columns[0])
            plt.ylabel(self.df.columns[1])
            plt.title(surface_function_names[count])
            plt.show()
            count += 1

        results_df['Model'] = surface_function_names
        results_df['R²'] = r2_list
        results_df['MAPE'] = MAPE_list
        results_df['AIC'] = aic_list

        results_df.sort_values(by=['R²'], ascending=False, inplace=True)

        results_df = results_df.round(2)
        print(results_df)
        results_df.to_csv('Surface_Model_Results.csv')

test = SurfaceModel('surface_chemistry_isotherms.csv')


def Arrhenius(T,Ea,A):
    R = 8.314
    return A * np.exp(-(Ea/R*T))

def Arrhenius_Linear(T,Ea,A):
    R = 8.314
    return np.log(A) - (Ea/R) * (1/T)

def convert_to_linear(x_vals,y_vals):
    y = np.log(y_vals)
    x = 1 / x_vals
    return x,y


class ArrheniusEquation:
    def __init__(self,filename,df=None,x_vals=None,y_vals=None):
        self.df = pd.read_csv(filename)

        self.x_vals = self.df.iloc[:, 0]
        self.y_vals = self.df.iloc[:, 1]

        self.x_vals, self.y_vals = convert_to_linear(self.x_vals,self.y_vals)

        print(self.df)
        print('=' * 50)

        results_df = pd.DataFrame()
        r2_list = []
        MAPE_list = []
        mse_list = []

        param, param_cov = curve_fit(Arrhenius_Linear, self.x_vals, self.y_vals)
        fitted_y = Arrhenius_Linear(self.x_vals, *param)

        r2 = r2_score(self.y_vals, fitted_y)
        mse = mean_squared_error(fitted_y, self.y_vals)
        mape = mean_absolute_percentage_error(self.y_vals, fitted_y)

        print(f'R² score: {r2:.2f}')
        print(f'MSE score: {mse:.2f}')
        print(f'MAPE score: {mape:.2f}%')
        print('=' * 50)

        r2_list.append(r2)
        MAPE_list.append(mape)
        mse_list.append(mse)

        print(f'Condition number of covariance matrix: {np.linalg.cond(param_cov):.2f}')
        print('High condition numbers indicate over parametrising, reduce the number terms.')

        for paramater in param:
            print(f'Paramater: {paramater:.3f}')
        print(param_cov)
        print('=' * 50)

        slope = (fitted_y.iloc[-1] - fitted_y.iloc[0]) / (self.x_vals.iloc[-1] - self.x_vals.iloc[0])
        Ea = -1 * slope * 8.314

        print(f'Activation energy (Ea) = {Ea:.2f}')
        print('='*50)

        plt.scatter(self.x_vals, self.y_vals, label=f'Actual Data')
        plt.plot(self.x_vals, fitted_y, label=f'Fitted Data: R² = {r2:.2f}', marker='x',color='#FF7A04')
        plt.legend()
        plt.xlabel('Temperature (1/K)')
        plt.ylabel('Ln(k)')
        plt.title('Arrhenius Equation')
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        ax1.scatter(self.x_vals**-1 - 273, np.exp(self.y_vals), label=f'Actual Data')
        ax2.plot(self.x_vals**-1, np.exp(fitted_y), label=f'Fitted Data: R² = {r2:.2f}', marker='x',color='#FF7A04')
        ax2.set_xlabel(self.df.columns[0])
        ax2.set_ylabel(self.df.columns[1])
        ax1.set_ylabel(self.df.columns[1])
        ax1.set_xlabel('Temperature (°C)')
        fig.legend(bbox_to_anchor=(-0.05, 0.33, 0.5, 0.5))
        plt.title('Arrhenius Equation')
        plt.tight_layout()
        plt.show()

        results_df['Model'] = ['Arrhenius Equation']
        results_df['R²'] = r2_list
        results_df['MAPE'] = MAPE_list
        results_df['MSE'] = mse_list

        results_df.sort_values(by=['R²'], ascending=False, inplace=True)

        results_df = results_df.round(2)
        print(results_df)
        results_df.to_csv('Arrhenius_Model_Results.csv')
        print('=' * 50)

        self.params = param

    def predict(self,x_to_predict):

        y_predicted = Arrhenius_Linear(x_to_predict, *self.params)
        y_predicted = np.exp(y_predicted)

        print(f'Predicted value: {y_predicted:.5f}')

        print('=' * 50)



test = ArrheniusEquation('arrhenius_dataset.csv')
test.predict(1/320)

