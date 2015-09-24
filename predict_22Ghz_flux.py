# Module that will predict the 22 GHz flux based on a model fit.

import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('/Users/ttshimiz/Github/bat-agn-sed-fitting/')


# Function to calculate the FIR flux using Equation 14 from Condon+1992
def calc_fir_flux(model):

    s60 = model.eval_grey(60.)
    s100 = model.eval_grey(100.)
    
    fir_flux = 1.26e-14*(2.58*s60 + s100)
    
    return fir_flux

    
# Function to calculate the 1.4 GHz flux based on the FIR flux and FIR-radio correlation
# from Eauation 15 of Condon+1992
def predict_1400mhz_flux(fir_flux, q=2.3):

    return 10**(np.log10(fir_flux/3.75e12) - q)
    

# Function to predict the 22 GHz flux by assuming a two component
# bremmstrahlung and synchrotron model.
def predict_22Ghz_flux(oneGHz_flux, alpha=0.1, gamma=0.8):

    # Decompose the flux at 1.4 GHz into thermal and
    # nonthermal using equation 5 of Condon+1992
    thermal_frac = 1+10*(1.4)**(alpha-gamma)
    therm_flux_1400 = oneGHz_flux/thermal_frac
    nontherm_flux_1400 = oneGHz_flux - therm_flux_1400
    
    # Extrapolate each component to 22 GHz
    therm_flux_22 = therm_flux_1400 / (1.4**(-alpha) / 22.**(-alpha))
    nontherm_flux_22 = nontherm_flux_1400 / (1.4**(-gamma) / 22.**(-gamma))
    total_22 = (therm_flux_22 + nontherm_flux_22) * 10**26
    
    return total_22
    

def run_prediction(model, q=2.3, alpha=0.1, gamma=0.8):

    fir_flux = calc_fir_flux(model)
    oneGHz_flux = predict_1400mhz_flux(fir_flux, q=q)
    predict_22 = predict_22Ghz_flux(oneGHz_flux, alpha=alpha, gamma=gamma)
    
    return predict_22
    