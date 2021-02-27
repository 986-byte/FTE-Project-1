import pandas as pd
import numpy as np

def euc_dis(v_1,v_2):    
    dis = 0    
    v_subt = np.subtract(v_1[:,:-1], v_2)   
    dis = np.linalg.norm(v_subt, 2, 1)    
    return dis

def knn(train, test, k):    
    distances = euc_dis(train, test)        
    idx = np.argpartition(distances, k)[:k]    
    output_values = [train[id, -1] for id in idx]    
    prediction = max(set(output_values), key=output_values.count)    
    return prediction

def import_file():    
    df_train = pd.read_csv (r'/Users/haper/Desktop/training.csv')    
    dt_train = df_train.to_numpy()

def main(k):    
    prediction = []    
    import_file()    
    preprocessing()    
    for samples in testing:        
        prediction.append(knn(train, test, k))    
    return prediction