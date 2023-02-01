from scipy import stats
import numpy as np
import pandas as pd


class DataPorcesser:
    def __init__(self, data, cols):
        self.max_len_ = 0
        self.data = data
        self.cols = cols
        self.created_cols = []


    def data_stats_func(self, data_):
        lengths_avg = []
        for i in data_:
            lengths_avg.append(len(i.split()))
        check_arr = np.array(lengths_avg)
        return stats.describe(check_arr)

    def data_stats_info(self):
        results = {}
        for i in self.cols:
            results[i] = self.data_stats_func(self.data[i])

        return results
            
    def selecting_max_length(self):
        """
            since this is paraphrasing the input and output should be same
            *   selecting the min value or (min -1) since eits the most appropriate value
        """
        max_req = []
        for i in self.data_stats_info():
            max_req.append(self.data_stats_info()[i].minmax[1])
        

        # return np.min(max_req)
        return np.min(max_req) -1


    # text shorter function processer (to apply to dataset)
    def shorter(self, text):
        text = str(text)
        text = text.split(' ')
        if(len(text) <= self.max_len_):
            return " ".join(text)
        else:
            return np.NaN        


    def data_shorting(self):
        """
            *   REMOVING ALL THE SENTENCES THAT HAS SENTENCES.LENGTH MORE THEN THE MAX_LEN_
            *   AND ADDING A NEW DATAFRAME 'short_clean_question'
            *   removing the raws which has empty elements
        """
        self.max_len_ = self.selecting_max_length()
        for i in range(len(self.cols)):
            self.created_cols.append(f'short_clean_{self.cols[i]}')
            
            self.data[f'short_clean_{self.cols[i]}'] = self.data[self.cols[i]].apply(self.shorter)

        # removing the raws which has empty elements
        for i in self.data.columns:
            if self.data.isna().sum().sum() > 0:
                self.data = self.data.dropna(subset=[i], how='all')
            else:
                return self.data


    # creating new data frame
    def proccessed(self):
        self.data_shorting()
        data_obj = {}
        for i in range(len(self.cols)):
            data_obj[self.cols[i]] = self.data[f'short_clean_{self.cols[i]}']
        
        new_data = pd.DataFrame(data_obj)


        # saving data.info
        f = open('./report/data_info.txt', 'w+')
        self.data.info(buf=f)
        f.close()

        # saving data.describe in both txet and csv formet
        self.data.describe().to_csv('./report/data_describe.txt', sep="\t")
        self.data.describe().to_csv('./report/data_describe.csv')


        # saving max length in a file
        f = open('./report/max_len_.txt','w')
        f.write('{}'.format(self.max_len_))
        f.close()

        # returning the result values
        return {'data':new_data, 'max_len_': self.max_len_}



