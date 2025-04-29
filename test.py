from Parameters import ParaManager_Unp, ParaManager_Pol
import numpy as np
import pandas as pd
import os
import csv
from Observables import GPDobserv


dir_path = os.path.dirname(os.path.realpath(__file__))

Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]

Para_Unp = ParaManager_Unp(np.array(Paralst_Unp[:-2]))
Para_Pol = ParaManager_Pol(np.array(Paralst_Pol))
Para_All = np.concatenate([Para_Unp, Para_Pol], axis=0)

def PDF_theo_s(x, t, Q, p, flv, Para, p_order):
    _PDF_theo = GPDobserv(x, 0, t, Q, p)
    return _PDF_theo.tPDF(flv, Para, p_order)

Para_spe = Para_All[0]

if __name__ == '__main__':
    
    os.makedirs(os.path.join(dir_path,"GUMP_Results_test"), exist_ok=True)
        
    x = np.exp(np.linspace(np.log(0.0001), np.log(0.05), 100, dtype = float))
        
    pdfglst = np.array([PDF_theo_s(x_i,0.,2.,1,'g',Para_spe, 2) for x_i in x ])
    
    pdfulst = np.array([x_i*PDF_theo_s(x_i,0.,2.,1,'u',Para_spe, 2) for x_i in x ])
    
    pdfdlst = np.array([x_i*PDF_theo_s(x_i,0.,2.,1,'d',Para_spe, 2) for x_i in x ])
    
    with open(os.path.join(dir_path,"GUMP_Results_test/Smallx_PDF_New.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,pdfulst,pdfdlst,pdfglst])) 
