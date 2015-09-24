# Script that calculates the predicted 22 GHz for the BAT AGN observed with the VLA

import numpy as np
import pandas as pd

import sys
sys.path.append('/Users/ttshimiz/Github/bat-agn-sed-fitting/')

import predict_22Ghz_flux as p2f

# Upload the names of the VLA sources
vla_names = np.loadtxt('bat_agn_vla_sources_names.txt', dtype=str)
bat_radio = pd.read_csv('/Users/ttshimiz/Github/bat-data/bat_20cm.csv', index_col=0)

# Directory where the best fit model pickle files are located
mod_dir = '/Users/ttshimiz/Github/bat-agn-sed-fitting/analysis/casey_bayes_results/beta_fixed_2_wturn_gaussianPrior/pickles/'

# Create a Pandas Series to store the predicted fluxes
result = pd.DataFrame(index=vla_names, columns=['22 GHz Predict', '1.4 GHz Predict'])

for n in vla_names:
    f = open(mod_dir+n+'_casey_bayes_beta_fixed_2_wturn_gaussianPrior.pickle', 'rb')
    fit_result = pickle.load(f)
    f.close()
    
    bf_mod = fit_result['best_fit_model']
    fflux= p2f.calc_fir_flux(bf_mod)
    oneGHz_predict_flux = p2f.predict_1400mhz_flux(fflux)
    predict_22 = p2f.predict_22Ghz_flux(oneGHz_predict_flux)
    
    result.loc[n, '22 GHz Predict'] = predict_22
    result.loc[n, '1.4 GHz Predict'] = oneGHz_predict_flux*10**26

result = result.join(bat_radio)
result.to_csv('predicted_22GHz_fluxes.csv', header=True, index_label='Name')