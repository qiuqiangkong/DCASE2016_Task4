'''
SUMMARY:  predict and write out results of evaluation data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from hat.optimizers import Rmsprop
from hat import serializations
from hat.metrics import prec_recall_fvalue
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data
import prepare_eva_data as pp_eva_data
import csv
import cPickle

# hyper-params
agg_num = 11
hop = 15
fold = 1
n_labels = len( cfg.labels )

# load model
md = serializations.load( cfg.eva_md_fd + '/md10.p' )

# prepare data
te_X = pp_eva_data.GetAllData( cfg.eva_fe_mel_fd, cfg.eva_csv_path, agg_num, hop )

# do recognize and evaluation
thres = 0.4     # thres, tune to prec=recall
n_labels = len( cfg.labels )

pp_dev_data.CreateFolder( cfg.eva_results_fd )
txt_out_path = cfg.eva_results_fd+'/task4_results.txt'
fwrite = open( txt_out_path, 'w')
with open( cfg.eva_csv_path, 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)

    # read one line
    for li in lis:
        na = li[1]
        full_na = na + '.16kHz.wav'
        
        # get features, tags
        fe_path = cfg.eva_fe_mel_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )

        p_y_pred = md.predict( X3d )
        p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
        
        # write out data
        for j1 in xrange(7):
            fwrite.write( full_na + ',' + cfg.id_to_lb[j1] + ',' + str(p_y_pred[j1]) + '\n' )
            
fwrite.close()
print "Write out to", txt_out_path, "successfully!"