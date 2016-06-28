'''
SUMMARY:  predict and write out results of evaluation data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from Hat.optimizers import Rmsprop
from Hat.metrics import prec_recall_fvalue
import Hat.backend as K
import config as cfg
import prepareData_eva as ppData_eva
import csv
import cPickle

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10
hop = 10
fold = 1
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md_eva/md100.p', 'rb' ) )

# prepare data
te_X = ppData_eva.GetAllData( fe_fd, cfg.eva_csv_path, agg_num, hop )

# do recognize and evaluation
thres = 0.4     # thres, tune to prec=recall
n_labels = len( cfg.labels )


fwrite = open('Results_eva/task4_results.txt', 'w')
with open( cfg.eva_csv_path, 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)

    # read one line
    for li in lis:
        na = li[1]
        full_na = na + '.16kHz.wav'
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )

        p_y_pred = md.predict( X3d )
        p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
        
        # write out data
        for j1 in xrange(7):
            fwrite.write( full_na + ',' + cfg.id_to_lb[j1] + ',' + str(p_y_pred[j1]) + '\n' )
            
fwrite.close()