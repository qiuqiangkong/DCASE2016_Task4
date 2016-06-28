'''
SUMMARY:  prepare data for evaluation
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import numpy as np
import csv
import cPickle
import prepareData as ppData
import config as cfg
from Hat.preprocessing import mat_2d_to_3d

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

if __name__ == "__main__":
    ppData.CreateFolder('Fe_eva')
    ppData.CreateFolder('Fe_eva/Mel')
    ppData.CreateFolder('Results_eva')
    ppData.CreateFolder('Md_eva')
    