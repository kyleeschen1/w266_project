
import numpy as np
import pandas as pd
import glob
from copy import copy 

#==============================================================================
# Formatting
#==============================================================================

def cc(lst, axis = 1):
    '''Concatenates two numpy arrays.'''
    return np.concatenate(lst, axis = axis)

def concat_all_files_into_one(path, name):
    '''Assemble all data into one'''
    files = glob.glob(path.format(name + "*"))
    files = [np.load(f) for f in sorted(files)]
    return np.concatenate(files) 

#==============================================================================
# Displays
#==============================================================================

def banner(text, symbol = "=", rep = 80):
    '''Prints a festive banner.'''
    print("\n{}\n{}\n{}\n".format(symbol * rep, text, symbol * rep))
    
def get_similarity(df1, df2, nrows = 10):
    '''Prints dot products of rows for two arrays.'''
    for i in list(range(nrows)):    
        print(np.dot(df1[i], df2[i].T))
        
#==============================================================================
# Label Encodings
#==============================================================================

def integer_encode(labels):
    cols = labels.unique()
    return pd.DataFrame([int(i) for row in labels for i, col in enumerate(cols)  if row == col])

def one_hot_encode(labels):
    cols = labels.unique()
    return pd.DataFrame({col : [(1 if row == col else 0) for row in labels] for col in cols})

#==============================================================================
# Calculations
#==============================================================================

def assess_accuracy(predictions, labels):
    '''Returns accuracy from probability dataframes.'''
    prediction_max_cols = np.argmax(predictions, axis=1)
    eval_max_cols = np.argmax(labels.to_numpy(), axis=1)
    return sum(prediction_max_cols == eval_max_cols) / labels.shape[0]
