# -*- coding: utf-8 -*-
import streamlit  as st
import FinanceDataReader as fdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

class AI():
    def __init__(self):
        st.title('종목 조회')
        
        STOCK_CODE =st.sidebar.text_input('매매할 종목코드를 입력해 주세요(005930)')  
        
        if st.sidebar.button('검색'):
            #종목 불러오기
            stock =self.fdr_(STOCK_CODE)
            st.dataframe(stock)
            
            #close 값 그리기
            dff =self.CloseDF_(stock)
            st.line_chart(dff)

            
            #정규화
            df ,scaler=self.MinMax_(stock)

            #numpy
            df=self.numpy_(df )

            #window
            window_size = 30
            X,Y =self.make_sequene_dataset(df, window_size)


            #split_
            x_train,y_train,x_test,y_test =self.split_(X,Y)


            #model
            model =self.model_(x_train)

            #EarlyStopping
            pred =self.early_(model,x_train,y_train,x_test,y_test)
            y_test=scaler.inverse_transform(y_test)
            pred=scaler.inverse_transform(pred)


            #시각화
            day = datetime.today()
            plus20_=pd.date_range(day, periods=20)
            df20_ = pd.DataFrame(pred, index=plus20_)
            df20_.columns=['pred']
            df20_=df20_.astype('int')
            ccc=pd.concat([dff,df20_[1:]])
            #st.dataframe(ccc)
            st.subheader('종가 예측')
            st.line_chart(ccc)
            st.write('당일 종가:',dff.iloc[-1].values[0])
            st.write('5일뒤 종가:',df20_.iloc[-15].values[0])
            st.write('20일뒤 종가:',df20_.iloc[-1].values[0])




    #종목 불러오기
    def fdr_(self,STOCK_CODE):
        stock = fdr.DataReader(STOCK_CODE,'2020')
        return stock
    
    def CloseDF_(self,stock):
        dff = pd.DataFrame(data=stock.Close, index=stock.index)
        return dff
    
    #정규화
    def MinMax_(self,stock):
        scaler = MinMaxScaler()
        scale_cols = ['Close']
        scaled = scaler.fit_transform(stock[scale_cols])
        df = pd.DataFrame(scaled, columns=scale_cols)
        return df, scaler

    # numpy
    def numpy_(self,df):
        df = df.to_numpy()
        return df

    #window
    def make_sequene_dataset(self,df, window_size):
        feature_list = []
        label_list = []
        for i in range(len(df)-window_size):
            feature_list.append(df[i:i+window_size])
            label_list.append(df[i+window_size])
        return np.array(feature_list), np.array(label_list)

    #split_
    def split_(self,X,Y):
        x_train = X[:-20]
        y_train = Y[:-20]
        x_test = X[-20:]
        y_test = Y[-20:]
        #x_train,x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=1, shuffle=False)
        return x_train,y_train,x_test,y_test

    #model
    def model_(self,x_train):
        model = Sequential()
        model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
        model.add(Dense(1, activation='linear'))
        #model.summary
        return model

    #EarlyStopping
    def early_(self,model,x_train,y_train,x_test,y_test):
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, callbacks=[early_stop])
        m=model.predict(x_test)
        return m
    
    
    
    
    
    
    
    
    
    
    
    
    
            #st.dataframe(dff)
            #st.write(bbb) #20일 시간
            #df20_=df20_.round(0)
            #st.dataframe(df20_)#pred dataframe
            #ccc=ccc.round(0)
            #st.table(ccc.info())
            #d=dff.iloc[-1].values[0]
            #ccc['pred'][-20]=d
    
#            result =self.result_(data, datas)    
            #data = pd.DataFrame(y_test)
            #datas = pd.DataFrame(pred)
'''
    #시각화
    def result_(self,data, datas):
        data.reset_index(drop=True, inplace=True)
        data.columns=['test']

        datas.reset_index(drop=True, inplace=True)
        datas.columns=['pred']
        result=pd.concat([data,datas],axis=1)#합친거

        return result
'''


'''
#index20_=time2 + pd.Timedelta(days=20)
 #time20_ = time2 + timedelta(days=20)

 #pred20_=pd.DataFrame(pred[0],index=dff.index+diff_days)
 #diff_days = datetime.timedelta(days=20)
 
# dff = pd.DataFrame(dff, index20_ )
 #dff=pd.DataFrame(dff, index=dff.index+diff_days)

 
 #aaa = pd.DataFrame(pred)
 #.dataframe(aaa, index=dff.index())
 
 
 #today = datetime.date.today()
 #dates = pd.date_range(today,today+20,freq='D')
 #result.index
 #st.line_chart(result)
 '''







