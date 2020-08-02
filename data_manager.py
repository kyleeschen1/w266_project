import numpy as np
from utilities import *
import json_lines as jsonl


COLS = ["sentence1", "sentence2", "concat"]
SETS = ["train", "dev", "test"]
FP = {"raw": "data/raw/snli_1.0_{}.jsonl",
      "bert": "data/bert_encodings/{}.npy",
      "use": "data/us_encodings/{}_us_embeddings.npy"}

class DataManager:
    
    def __init__(self, max_rows = None):
        self.max_rows = max_rows
    
    def load_all_labels(self, by_annotator = False):
        '''Returns all labels in a dictionary.'''
        return {s: self.__load_labels(s, by_annotator) for s in SETS}

    def load_all_data(self, name, concat = False):
        '''Returns all data in a dictionary.'''
        fp = FP[name]
        return {s : {col : self.__load_data(fp, s, col, concat) for col in COLS} for s in SETS}
    
    #==============================================================================
    # Private class helpers
    #==============================================================================
            
    def __load_data(self, fp, setname, name, concat = False):
        name = setname + "_" + name
        if concat:
            df = concat_all_files_into_one(fp, name)
        else:
            df = np.load(fp.format(name))
        return self.__filter_max_rows(df)
           
    #==============================================================================
    # Label Load Helpers
    #==============================================================================

    def __load_labels(self, file_name, by_annotator):
        '''Loads and formats labels.'''

        # load file
        with open(FP["raw"].format(file_name), "r") as f:
            data = [DataManager.format_label_cols_only(row) for row in jsonl.reader(f)] 

        # convert to dataframe, and eliminate unlabeled rows
        data = pd.DataFrame(data)
        data = data[data["gold_label"] != "-"]
        data = self.__filter_max_rows(data)
        
        if by_annotator:
            return {col : DataManager.one_hot_formatting(data[col]) for col in data.columns}
        else:
            return DataManager.one_hot_formatting(data["gold_label"])
        
    @staticmethod
    def format_label_cols_only(row):
        '''Special formatting for files.'''  
        row.pop("annotator_labels")
        return row
            
    @staticmethod
    def format_label_cols(row, by_annotator):
        '''Special formatting for files.'''  
        if row["gold_label"] != "-":
            labels = row.pop("annotator_labels")
            if by_annotator:
                labels = {"a{}".format(i + 1): labels[i] for i in range(len(labels))}
                labels.update({"majority_label": row.pop("gold_label")})
            else:
                labels = {"gold_label": row["gold_label"]}
            return labels

    @staticmethod
    def one_hot_formatting(col):
        '''Pivots to one-hot-encoding, then drops null column.'''
        col = one_hot_encode(col)
        return col.loc[:, ["neutral", "entailment", "contradiction"]]
    
    #==============================================================================
    # General Helpers
    #==============================================================================
    
    def __filter_max_rows(self, df):
        '''Filters for max rows if dataframe is larger than set limit.'''
        if (self.max_rows is not None) and (df.shape[0] > self.max_rows):
            df =df[:self.max_rows]
        return df
        
        
#==============================================================================
# Build Dataframes over Data Dictionaries
#==============================================================================
    
def build_over_dict(data_dict, func):
    return {s: func(data) for s, data in data_dict.items()}

def build_concat_sentences(data):
    p1 = cc((data["sentence1"], data["sentence2"]))
    p2 = cc((data["sentence2"], data["sentence1"]), axis = 1)
    return cc([p1, p2], 0)

def build_difference_sentences(data):
    p1 = data["sentence1"] - data["sentence2"]
    #p2 = data["sentence2"] - data["sentence1"]
    #return cc([p1, p2], 0)
    return p1

