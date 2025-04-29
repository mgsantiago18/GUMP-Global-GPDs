### Module for calculating meson production cross-sections and TFFs

import numpy as np
import pandas as pd
import Evolution as ev
import Observables as obs
import os
import matplotlib.pyplot as plt
from iminuit import Minuit
from numpy import cos as Cos
from numpy import sin as Sin
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate
from scipy.integrate import quad

from numba import njit, vectorize



Minuit_Counter = 0

Time_Counter = 1

dir_path = os.path.dirname(os.path.realpath(__file__))

"""

***************************** Masses, decay constants, etc. ***********************

"""
M_p = 0.938
M_n = 0.940
M_rho = 0.775
M_phi = 1.019
M_jpsi = 3.097
gevtonb = 389.9 * 1000
alphaEM = 1 / 137.036



"""

******* R Ratio (longitudinal/transverse separation) Parametrization and Fits ************

"""


# 1) Loading the combined H1 and ZEUS R‐ratio data for ρ meson:
#    • We’ve taken both the ZEUS and H1 measurements, merged them into one table,
#      and now we are fitting a single parametrization to the combined HERA data.

RrhoZEUSnH1= pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1.csv'), header = None, 
             names = ['Q', 'R','stat_pos','stat_neg','syst_pos','syst_neg'] , dtype = {'Q': float, 'R': float, 'stat_pos': float,'stat_neg': float, 'syst_pos':float, 'syst_neg':float})

# 2) Converting the Q² values in the file to Q by taking the square root.
#    This matches our convention: we want Q (GeV) not Q².

RrhoZEUSnH1['Q'] = np.sqrt(RrhoZEUSnH1['Q'])

# 3) Defining R(Q,a,p,meson): the ratio σ_L / σ_T for meson production.
#    This follows Eq.(32) in arXiv:1112.2597 for ρ, but here we handle
#    both ρ (meson==1) and J/ψ (meson==3) cases in one function.


def R(Q:float, a:float, p:float, meson:int):
    """ The R ratio: Longitudinal DVMP cross section/ Transverse DVMP cross section
    
    Args:
       Q: The photon virtuality 
       a: Parameter
       p:Parameter
       meson: 1 for rho, 3 for jpsi, 2 is saved for phi to use later
    
    Returns: The parametrization of R factor of the  L/T separation  as in  Eq.(32) in https://arxiv.org/pdf/1112.2597"
    
    """
    if (meson==1): 
        return (Q**2 / M_rho**2) * (1 + np.exp(a) * Q**2 / M_rho**2) ** (-p)
    if (meson==3): 
        return  (Q**2/M_jpsi**2)



# 4) Extracting Q values and experimental R values from the ρ production data (H1 and Zeus combined)
  
Q_vals = RrhoZEUSnH1['Q'].values
R_exp_rho  = RrhoZEUSnH1['R'].values

N = len(R_exp_rho) # Number of experimental points



# Extracting the statistical and systematic errors from the data
    
stat_errors_pos=RrhoZEUSnH1['stat_pos'].values
stat_errors_neg=RrhoZEUSnH1['stat_neg'].values
syst_errors_pos=RrhoZEUSnH1['syst_pos'].values
syst_errors_neg=RrhoZEUSnH1['syst_neg'].values


# Combining +/– errors in quadrature to get symmetric stat & syst errors,
# then combining those to get a total uncertainty:
    
stat_errors = np.sqrt(stat_errors_pos**2 + stat_errors_neg**2)/np.sqrt(2)
syst_errors = np.sqrt(syst_errors_pos**2 + syst_errors_neg**2)/np.sqrt(2)
tot_errors  = np.sqrt(stat_errors**2 + syst_errors**2)


# 5) Defining the chi² cost function for fitting a, p to the ρ–data:

def R_rho_cost(a, p):
    """
    Cost function for R(Q) using H1 and Zeus data combined.
    
    Parameters:
      
      a, p  : Free parameters in the model
      
    Returns:
      Reduced chi2: (Sum of squared differences between model prediction and experimental data divided by the total errror)/Degrees of freedom
      
    """

    k=2 # Number of parameters in the parametrization, 2 for a and p
    
    dof=N-k # Degrees of freedom = Number of experimental points - Number of parameters in the parametrization
    
    R_pred = R(Q_vals, a, p,meson=1)   # Computing the model prediction for each Q value for ρ meson
      
    chi2 = np.sum(((R_exp_rho - R_pred)/tot_errors) ** 2)  # Standard χ² sum over data points
    
    chi2_red = chi2 /dof #Reduced χ²:  Reduced χ² = χ²/dof
    
    return chi2_red


# 6) Using iminuit to minimize the cost function and extract best–fit values:

m = Minuit(R_rho_cost, a=2.5, p=0.7) # initial guesses.

m.migrad()  # run the minimizer
m.hesse()   # compute the uncertainties via the Hessian


# Pulling out fitted parameters and their 1σ errors:
val_a = m.values['a']
val_p = m.values['p']
var_a = m.errors['a']
var_p = m.errors['p']

# Correlation between a & p from the off–diagonal of the covariance matrix:
corr_ap = m.covariance[0,1] 


#Print the covariance matrix
print("Covariance matrix:")
print(m.covariance)

print("Fitted parameters:")
print(f"  a = {val_a:.4f} ± {var_a:.2f}")
print(f"  p = {val_p:.4f} ± {var_p:.2f}")
print(f"  correlation = {corr_ap:.2f} ")



#Calculating central, upper, and lower predictions for the model for plotting or further error propagation, 
Q_fit = np.linspace(min(RrhoZEUSnH1['Q']), max(RrhoZEUSnH1['Q']))
R_fit_central = R(Q_fit, val_a, val_p,meson=1)


# -------------------------------------------------------------------
# Propagating the uncertainties in (a, p) through R(Q) via standard error
# propagation, then plotting the fitted curve with its ±1σ “error band.”
# -------------------------------------------------------------------

# 1) Computing σ[R] based on variances of a and p and the correlation between them

def stan_dev_R_rho(Q, a, p):
    
    partial_derivative_a=-p * (Q**2 / M_rho**2)**2 * np.exp(a) * (1 + np.exp(a) * Q**2 / M_rho**2) ** (-p-1)
    partial_derivative_p=-(Q**2 / M_rho**2) * (1 + np.exp(a) * Q**2 / M_rho**2)**(-p) * np.log(1 + np.exp(a) * Q**2 / M_rho**2)
    
    part_a = partial_derivative_a**2 * var_a**2
    part_p = partial_derivative_p**2 * var_p**2
    part_ap =partial_derivative_a*partial_derivative_p*corr_ap
    variance_R_rho=part_a + part_p + 2 * part_ap
    #print(variance_R_rho)
    return np.sqrt(variance_R_rho)

"""

# 2) Building an array of σ[R] over the fit grid Q_fit:
    
stan_dev_R_rho = np.array([stan_dev_R_rho(Q, val_a, val_p) for Q in Q_fit])

# 3) Upper & lower error bands:
    
R_upper_rho = R_fit_central + stan_dev_R_rho
R_lower_rho= R_fit_central - stan_dev_R_rho    
    
# 4) Plotting the central fit, the ±1σ band, and the data points:  
    
plt.figure(figsize=(10, 6))
plt.plot(Q_fit, R_fit_central, label='R(Q^2)', color='blue')
plt.fill_between(Q_fit, R_lower_rho, R_upper_rho, color='blue', alpha=0.2, label='Error Band')
plt.errorbar(Q_vals, R_exp_rho, yerr=[tot_errors], fmt='bo', label="Experimental Data", capsize=5)

plt.title('R(Q) with Variance Error Bands')
plt.xlabel('$Q^2$ (GeV$^2$)')
plt.ylabel('R($Q^2$)')
plt.legend()
plt.grid(True)
plt.show()
    
    
 """   
    
"""

******************************Cross-sections for proton target (currently for virtual photon scattering sub-process)*********************************

"""



def epsilon(y:float):
    """ Photon polarizability.

    Args:
       y (float): Beam energy lost parameter
     

    Returns:
        epsilon:  "Eq.(31) in https://arxiv.org/pdf/1112.2597" 
    """
    return (1 - y) / (1 - y + y**2 / 2)



     
def MassCorr(meson:int):
    """ Mass corrections 

     Args:
         meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
      

     Returns:
        mass correction only for j/psi
     """
  
    if (meson==3):
        return  M_jpsi
    else:
        return 0
  
         

    
    #return Q**2/M_meson(meson)**2 /(1 + a_meson(meson) * Q**2 / M_meson(meson)**2)**p_meson(meson)


# -----------------------------------------------------------------------------
# dsigmaL_dt : longitudinal differential cross section (in t)
# -----------------------------------------------------------------------------
@np.vectorize
def dsigmaL_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex):
    """Longitudinal DVMP cross section differential only in t
          
      Args:
          y (float): Beam energy lost parameter
          xB (float): x_bjorken
          t (float): momentum transfer square
          Q (float): photon virtuality
          TFF (complex): Transition form factor H 
          ETFF (complex): Transition form factor E
          MassCorr(int): Mass corrections to the cross section.  Nonzero only for j/psi
   
      

      Returns:
          
          Eq.(2.8) as in https://arxiv.org/pdf/2409.17231"
    """

    return  ( 4* np.pi**2  *alphaEM * xB ** 2 / ((Q**2 + MassCorr(meson)**2) ** 2)) * (Q/ (Q**2 + MassCorr(meson)**2)) ** 2 * (Real(HTFF* Conjugate(HTFF)) - t/4/ M_p**2 * Real(ETFF* Conjugate(ETFF)))
                                         
                                                
        
# return  gevtonb *2*np.pi**2 * alphaEM * (xB ** 2 / (1 - xB) * Q**4 * np.sqrt(1 + eps(xB, Q) ** 2))* Cunp(xB, t, Q, HTFF_rho, ETFF_rho, meson)






# -----------------------------------------------------------------------------
# dsigma_dt : total differential cross section (only in t)
# -----------------------------------------------------------------------------

@np.vectorize
def dsigma_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex,a:float,p:float):
    """The total DVMP cross section differential only in t
          
      Args:
          y (float): Beam energy lost parameter
          xB (float): x_bjorken
          t (float): momentum transfer square
          Q (float): photon virtuality
          TFF (complex): Transition form factor H 
          ETFF (complex): Transition form factor E
          MassCorr(int): Mass corrections to the cross section.  Nonzero only for j/psi
   
      

      Returns:
          
          Eq.(2.16) as in https://arxiv.org/pdf/2409.17231"
    """

    return  gevtonb *dsigmaL_dt(y, xB, t, Q, meson, HTFF, ETFF)*(epsilon(y)+1/R(Q,a,p,meson))
 