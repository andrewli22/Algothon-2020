from model import create_model
from model import test_predictions
import pandas as pd
import numpy as np


model = create_model("SPY.csv")
predictions, etf_prices = test_predictions(model,"SPY-5-years.csv")
p_list = predictions.tolist()
balance = 1000000
stock_count = 0
etf_price_list = etf_prices.tolist()


for i in range(0,len(p_list)):
    prediction = p_list[i][0] 
    if (prediction < 0.005):
        balance += stock_count*etf_price_list[i]
        stock_count = 0
    elif (prediction < 0.01 and prediction > 0.005):
        stock_count += 50
        balance -= etf_price_list[i]*50
    elif (prediction > 0.01):
        stock_count += 100
        balance -= etf_price_list[i]*100

balance += etf_price_list[-1]*stock_count 
print("P\L = " + str(balance-1000000))