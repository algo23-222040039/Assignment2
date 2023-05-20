# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:38:26 2023

@author: lijia
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime,date,time






def RSI(data,n):
    close = data['close']
    low = data['low']
    high = data['high']
    change = data['chg']
    
    #n天上涨或下跌点数：1.首先构造上涨序列和下跌序列；2.取rolling计算平均值
    change_p = change.apply(lambda x: x if x>0 else 0)
    change_n = change.apply(lambda x: -x if x< 0 else 0)
    change_p2 = change.apply(lambda x: x if x>0 else None)
    change_n2 = change.apply(lambda x: -x if x< 0 else None)
    
    
    
    rolling_low = low.rolling(n)
    rolling_high = high.rolling(n)
    rolling_close = close.rolling(n)#n天的
    # mean_chg_p = change_p.rolling(n).mean() / change_p2.rolling(n).count()
    # mean_chg_n = change_n.rolling(n).mean() / change_n2.rolling(n).count()
    mean_chg_p = change_p.rolling(n).mean()
    mean_chg_n = change_n.rolling(n).mean() 
  
    RS = mean_chg_p/mean_chg_n
    RSI = 100 - (100/(1+RS))
    data['RSI'] = RSI
   
    return data
    
    



#计算信号
def signal2(data):
    RSI = data['RSI']
    signals = []
    
    
    K1 = 80
    K2 = 60
    K3 = 30
    K4 = 20

    
    i = 0
    for pre_RSI, RSI in zip(RSI.shift(1),RSI):
        
        signal = None
        
        # if pre_RSI < K3 and RSI > K3: #超卖回归中间行情，卖出
        #     signal = -1
        # elif pre_RSI > K3 and RSI <K3: #向下进入超卖区间，买入
        #     signal = 1
        # # elif pre_RSI < K3 and RSI > K3:
        # #     signal = -1

        if pre_RSI <K2 and RSI >K2: #走入过热区间，买入
            signal = 1
        elif pre_RSI >K2 and RSI < K2:#向下进入正常区间，卖出
            signal = -1

            
        signals.append(signal)
    
    data['signal'] = signals
    return data

#计算持仓
def position(data):
    data['signal_last'] = data['signal'].shift(1)
    data['position'] = data['signal'].fillna(method='ffill').shift(1).fillna(0)
    return data



def date2row(date,dateSeries, type = "yyyy-mm-dd",):
    
    if type == "yyyy-mm-dd":
        d = datetime.strptime(date,'%Y-%m-%d')
        
    index_date = 0
    for i in range(0,np.size(dateSeries)):
        
        if dateSeries[i] <= d:
            index_date = index_date + 1
        else:
            break
        
    return index_date





    
def statistic_performance1(data, initCurrency = 10000, fee =0.00035):
    
    position = data['position'] # 获得交易方向
    high = data['high']
    low = data['low']
    avg = high*0.5+low*0.5
    
    data_period = (data['日期'].max()-data['日期'].min()).days
    r0 = 0
    currency = []
    hold = []
    value = []
    
    #生成持仓数据：无卖空
    i =0
    for posi, pri in zip(position,avg):
        if i ==0:
            c = initCurrency
            h = 0
        else:
            c = currency[i-1]
            h = hold[i-1]
            
        p = pri #指数价格
        if posi == 1 :
            h = h + c/ (1+fee)/p
            c = 0
            #转换为多头
        elif posi ==-1 :
            c = c + h * p *(1-fee)
            h = 0
        currency.append(c)
        hold.append(h)
        value.append(h * p +c)
        i = i+1
        
        
    data['value']=value
    data['hold']=hold
    data['currency']=currency
    
    hold_r = data['value']/data['value'].shift(1) -1
    hold_win = hold_r>0
    hold_cumu_r = (1+hold_r).cumprod() - 1
    drawdown = (hold_cumu_r.cummax()-hold_cumu_r)/(1+hold_cumu_r).cummax()  
    data['hold_r'] = hold_r
    data['hold_win'] = hold_win
    data['hold_cumu_r'] = hold_cumu_r
    data['drawdown'] = drawdown
    v_hold_cumu_r = hold_cumu_r.tolist()[-1]
    v_pos_hold_times= 0 
    v_pos_hold_win_times = 0
    v_pos_hold_period = 0
    v_pos_hold_win_period = 0
    v_neg_hold_times= 0 
    v_neg_hold_win_times = 0
    v_neg_hold_period = 0
    v_neg_hold_win_period = 0
    tmp_hold_r = 0
    tmp_hold_period = 0
    tmp_hold_win_period = 0
   
    for w, r, pre_pos, pos in zip(hold_win, hold_r, position.shift(1), position):
        # 有换仓（先结算上一次持仓，再初始化本次持仓）
        if pre_pos!=pos: 
            # 判断pre_pos非空：若为空则是循环的第一次，此时无需结算，直接初始化持仓即可
            if pre_pos == pre_pos:
                # 结算上一次持仓
                if pre_pos>0:
                    v_pos_hold_times += 1
                    v_pos_hold_period += tmp_hold_period
                    v_pos_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:
                        v_pos_hold_win_times+=1
                elif pre_pos<0:
                    v_neg_hold_times += 1      
                    v_neg_hold_period += tmp_hold_period
                    v_neg_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:                    
                        v_neg_hold_win_times+=1
            # 初始化本次持仓
            tmp_hold_r = 0
            tmp_hold_period = 0
            tmp_hold_win_period = 0  
        else: # 未换仓
            if abs(pos)>0:
                tmp_hold_period += 1
                if r>0:
                    tmp_hold_win_period += 1
                if abs(r)>0:
                    tmp_hold_r = (1+tmp_hold_r)*(1+r)-1       

    v_hold_period = (abs(position)>0).sum()
    v_hold_win_period = (hold_r>0).sum()
    v_max_dd = drawdown.max()    
    #年化收益 =总收益/
    v_annual_ret = pow( 1+v_hold_cumu_r, 
                      365/data_period)-1
    v_annual_std = hold_r.std() * np.sqrt(250*1440/data_period) 
    v_sharpe = v_annual_ret / v_annual_std
    performance_cols = ['累计收益', 
                        '多仓次数', '多仓胜率', '多仓平均持有期(交易日)', 
                        '空仓次数', '空仓胜率', '空仓平均持有期(交易日)', 
                        '日胜率', '最大回撤', '年化收益/最大回撤',
                        '年化收益', '年化标准差', '年化夏普']
    performance_values = ['{:.2%}'.format(v_hold_cumu_r),
                          v_pos_hold_times, 
                          '{:.2%}'.format(v_pos_hold_win_times/v_pos_hold_times), 
                          '{:.2f}'.format(v_pos_hold_period/v_pos_hold_times),
                          v_neg_hold_times, 
                          '{:.2%}'.format(v_neg_hold_win_times/v_neg_hold_times), 
                          '{:.2f}'.format(v_neg_hold_period/v_neg_hold_times),
                          '{:.2%}'.format(v_hold_win_period/v_hold_period), 
                          '{:.2%}'.format(v_max_dd), 
                          '{:.2f}'.format(v_annual_ret/v_max_dd),
                          '{:.2%}'.format(v_annual_ret), 
                          '{:.2%}'.format(v_annual_std), 
                          '{:.2f}'.format(v_sharpe)]
    performance = pd.DataFrame(performance_values, index=performance_cols)
    return data, performance
    
    
    

'''
下面开始回测
'''

#读取数据
data0=pd.read_excel('BTC-USD.xlsx')
str1 = 'BTC'

# data0=pd.read_excel('000300.xlsx')
# str1 ='HS300'


data=data0.copy()
#对列名重命名，便于后面计算
data.rename(columns={'开盘价(元)':'open','最高价(元)':'high','最低价(元)':'low','收盘价(元)':'close','成交额(百万元)':'vol'},inplace=True)
#计算每日涨跌幅

data['pre_close'] = data['close'].shift(1)
data['pct_chg'] = (data['close']-data['pre_close'])/data['pre_close']
data['chg'] = data['close']-data['pre_close']
#生成因子序列
data_RSI = RSI(data, 14)
#生成信号序列和持仓序列
data_signal = signal2(data_RSI)
data_position = position(data_signal)

#回测变量：
date1 = "2018-01-01"
date2 = "2022-12-31"
fee = 0.00035



data_test = data_position.iloc[date2row(date1,data['日期']):date2row(date2,data['日期'])]


result, performance = statistic_performance1(data_test)
print(performance)



#结果可视化
plt.figure(figsize=(20,8))
plt.title('RSI Performance')
cumu_hold_close = (result['hold_cumu_r']+1)*data_test['close'].tolist()[0]
plt.plot(data_test['日期'],cumu_hold_close,color='black')
plt.plot(data_test['日期'],data_test['close'],color='red')
plt.legend(['RSI',str1])
plt.grid()
plt.show()

plt.figure(figsize=(20,8))
plt.title('Drawdown')
plt.plot(data_test['日期'],-result['drawdown'],color='black')
# plt.ylim([-0.8,0])
plt.grid()
plt.show()




