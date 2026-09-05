import numpy as np
from scipy.optimize import curve_fit, minimize, fsolve

# Hendersson-Hasselbach equation
def HH(pKa, A, HA):

    pH = pKa + np.log10(A / HA)
    return pH

class pHBuffer:
    def __init__(self, pH, pKa, pKb):
        self.pKa = pKa
        self.pH = pH
        self.pKb = pKb

        if self.pKb is not None:
            self.pKa = 14 - self.pKb


    def find_optimal(self):

        # use to find the amoount of A- / HA needed
        def HH_optimise_x(params, pH=self.pH, pKa=self.pKa):
            x = params
            # x = A- / HA
            return pH - (pKa + np.log10(x))

        p0 = [1]
        res = fsolve(HH_optimise_x, p0)
        res = res[0]

        print(f'To achieve a buffer pH of {self.pH}: Need a pKa of {self.pKa} and a [A-] / [HA] ratio of {res:.2f}')
        if res < 1:
            print('Your acid must be in excess!')
        if res > 1:
            print('Your base must be in excess!')
        if res == 1:
            print('Your acid and base must be equal!')

    def get_pH(self, A, HA):
        self.A = A
        self.HA = HA

        pH = HH(self.pKa, self.A, self.HA)
        print(f'Calculated pH based on HH: {pH:.2f}')
