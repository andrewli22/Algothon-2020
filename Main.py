from model import create_model
from model import test_predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
data = "SPY-5-years.csv"
model = create_model(data)
predictions, etf_prices = test_predictions(model,"SPY.csv")
p_list = predictions.tolist()
balance = 1000000
stock_count = 0
etf_price_list = etf_prices.tolist()
pl = []
days = []

for i in range(0,len(p_list)):
    prediction = p_list[i][0] 
    if (prediction < 0.005):
        balance += stock_count*etf_price_list[i]
        stock_count = 0
    elif (prediction < 0.01 and prediction > 0.005):
        stock_count += 25
        balance -= etf_price_list[i]*25
    elif (prediction > 0.01):
        stock_count += 50
        balance -= etf_price_list[i]*50
    pl.append(((balance+etf_price_list[i]*stock_count)-1000000))
    days.append(i)


plt.plot(days,pl)
balance += etf_price_list[-1]*stock_count 
print("P\L = " + str(balance-1000000))
plt.show()
