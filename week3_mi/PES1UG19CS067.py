'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random
import math


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    #TODO
    no_of_columns=df.shape[1]
    no_of_rows=df.shape[0]
    name=df.columns[no_of_columns-1]
    total = df.groupby(name).count()
    # pos=total.iloc[0][0]
    # neg=total.iloc[1][0]
    entropy=0
    for i in range(total.shape[0]):
        value=total.iloc[i][0]
        value=value/no_of_rows
        entropy=entropy-(math.log(value,2))*(value)
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    #TODO
    att_067=df[attribute].unique()
    # print(type(att_067))
    # print(att_067)
    avg_info=0
    no_of_rows=df.shape[0]
    for i in att_067:
        entropy_067=get_entropy_of_dataset(df.loc[df[attribute] == i])
        total_067=df.loc[df[attribute] == i].count()
        avg_info=avg_info+ (entropy_067*total_067/no_of_rows)
        #print(info)
    #print(info[0])
    return avg_info[0]


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    information_gain=get_entropy_of_dataset(df) - get_avg_info_of_attribute(df,attribute)
    return information_gain

#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    dictionary_067=dict()
    ans_067=list()
    no_of_columns=df.shape[1]
    att_067=df.columns[0]
    for i in df.columns:
        if(i==df.columns[no_of_columns-1]):
            break
        dictionary_067[i]=get_information_gain(df,i)
        if(dictionary_067[i]>dictionary_067[att_067]):
            att_067=i
    ans_067.append(dictionary_067)
    ans_067.append(att_067)
    ans_067=tuple(ans_067)
    return ans_067

