'''
SUMMARY:  prepare data for evaluation
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import numpy as np
import csv
import cPickle
import prepare_dev_data as pp_dev_data
import config as cfg
from hat.preprocessing import mat_2d_to_3d
import os

def GetAllData( fe_fd, csv_file, agg_num, hop ):
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        Xlist.append( X3d )

    return np.concatenate( Xlist, axis=0 )
    
# size: n_songs*n_chunks*agg_num*n_in
def GetEvaSegData( fe_fd, agg_num, hop ):
    te_Xlist = []
        
    names = os.listdir( fe_fd )
    te_na_list = []
        
    # read one line
    for na in names:
        fe_path = fe_fd + '/' + na
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        te_Xlist.append( X3d )
        te_na_list.append( na[0:-2] )

    return np.array( te_Xlist ), te_na_list

if __name__ == "__main__":
    pp_dev_data.CreateFolder( cfg.eva_fe_fd )
    pp_dev_data.CreateFolder( cfg.eva_fe_mel_fd )
    pp_dev_data.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=0 )
    