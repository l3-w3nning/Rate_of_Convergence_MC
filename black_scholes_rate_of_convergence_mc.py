import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
random.seed(0)
import sqlite3
from scipy.stats import norm
from os import getcwd




def optionval(strike, stockprice):
    return max([0,stockprice-strike])

def riskneutralprocess(sigma,deltat,S0,rfr):
    return S0*np.exp((rfr-sigma**2/2)*deltat+sigma*np.sqrt(deltat)*random.gauss(0,1))

    


def bs_callprice(sigma,deltat,strike,S0,rfr):
    d1=(np.log(S0/strike)+(rfr+sigma**2/2)*deltat)/(sigma*np.sqrt(deltat))
    d2=d1-sigma*np.sqrt(deltat)
    return S0*norm.cdf(d1)-norm.cdf(d2)*strike*np.exp(-rfr*deltat)

def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    
    vega = S  * np.sqrt(T) * norm.pdf(d1)
    return vega

def implicit_vola(deltat,strike,S0,rfr,optionprice,tol=0.0001,max_iterations=1000):
    
    '''
    :param S0: Asset price
    :param strike: Strike price
    :param deltat: Time to Maturity
    :param rfr: risk-free rate (treasury bills)
    :param optionprice: optionprice
    :return: implicit vola according to bs model with newton raphson method
    '''

    #the function runs into problems when y-axis is breached - therefore limit sigma to small values larger than 0
    sigma=np.sqrt(2*np.pi/deltat)*optionprice/S0
    
    for i in range(max_iterations):
        diff=bs_callprice(sigma,deltat,strike,S0,rfr)-optionprice
        
        if abs(diff)<tol:
            print('end of iteration, method converged, implicit vola reads',sigma)
            break
        if sigma-diff/vega(S0,strike,deltat,rfr,sigma)>0:
            sigma=sigma-diff/vega(S0,strike,deltat,rfr,sigma)
    return sigma



    

def get_price_eurocall_mc(numsims,sigma,deltat,strike,S0,rfr):

    prices_numsims=np.array([])
    for j in range(numsims):
        prices=np.array([])
        for i in range(j):
            randomval=riskneutralprocess(sigma,deltat,S0,rfr)
            prices=np.append(prices,optionval(strike,randomval))
        prices=np.array(prices)
        optionprice=np.exp(-rfr*deltat)*np.average(prices)
        prices_numsims=np.append(prices_numsims,optionprice)
        print(j)
    
    return prices_numsims


def get_price_eurocall_fixed_n(numsims,sigma,deltat,strike,S0,rfr):

    prices_numsims=np.array([])
    for j in range(numsims):
            randomval=riskneutralprocess(sigma,deltat,S0,rfr)
            prices_numsims=np.append(prices_numsims,optionval(strike,randomval))
    
    optionprice=np.exp(-rfr*deltat)*np.average_numsims(prices_numsims)
        
    
    return optionprice

def main():
    #establish connection to database and proceed to read input data from db
    pwd = getcwd()
    dbconn=sqlite3.connect(pwd+'/optionpricing.db')
    curs = dbconn.cursor()
    results=[]
    colnames=[]
    for row in dbconn.execute("SELECT * FROM optionpricing ORDER BY oid DESC LIMIT 1"):  
        results.append(row)
    results=np.array(results[0])
    for col in dbconn.execute("PRAGMA table_info(optionpricing)"):
        colnames.append(col)
    dbconn.commit()
    dbconn.close()
    colnames=np.array(colnames)

    colnames=colnames[:,1]
    data=dict(zip(colnames,results))

    data_df=pd.DataFrame(data,index=[0])
    data_df = data_df.astype({'numsims':'int','bid':'float','ask':'float','RiskFreeRate':'float','Deltat':'float','S0':'float','Strike':'float','contractSymbol':'string'})
    print(data_df.dtypes)
    print(data_df)

    numsims = 1000
    deltat=data_df.loc[0,"Deltat"]
    strike=data_df.loc[0,"Strike"]
    s0=data_df.loc[0,"S0"]
    optionprice=data_df.loc[0,"ask"]
    rfr=data_df.loc[0,"RiskFreeRate"]
    
    imp_vola=implicit_vola(deltat,strike,s0,rfr,optionprice)

    fig = plt.figure()

    plt.subplot(1,2,1)
    prices_call=get_price_eurocall_mc(numsims=numsims,sigma=imp_vola,deltat=deltat,strike=strike,S0=s0,rfr=rfr)    
    plt.plot(prices_call)
    plt.xlabel("Number of Simulations")
    plt.ylabel("Simulated Call Price")
    plt.title("Convergence of Monte-Carlo Simulation with N")
    plt.axhline(y=bs_callprice(imp_vola,deltat,strike,s0,rfr),c="r",linestyle="-",label="Exact BS-Price")
    plt.plot()

    range_numsims = np.arange(1,numsims)
    rate_of_convergence = [abs(prices_call[1]- bs_callprice(imp_vola,deltat,strike,s0,rfr))/np.sqrt(x) for x in range_numsims]
    
    plt.subplot(1,2,2)
    plt.loglog(abs(prices_call-bs_callprice(imp_vola,deltat,strike,s0,rfr)),label="Absolute Error MC Price/BS Price")
    plt.loglog(rate_of_convergence,c="red",linestyle="-",label="Rate of Convergence (1/N^(0.5))  ")
    plt.legend()
    plt.xlabel("Number of Simulations")
    plt.ylabel("Absolute Error")
    plt.title("Rate of Convergence MC Price")
    plt.show()


if __name__ == "__main__":

    main()





