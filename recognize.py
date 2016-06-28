'''
SUMMARY:  Dcase 2016 Task 4. Audio Tagging
          Recognize and evaluate the f value or EER
AUTHOR:   Qiuqiang Kong
Created:  2016.05.29
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
import prepareData as ppData
import csv
import cPickle

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10
hop = 10
fold = 1
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md/md100.p', 'rb' ) )

# prepare data
_, _, te_X, te_y = ppData.GetAllData( fe_fd, agg_num, hop, fold )

# do recognize and evaluation
thres = 0.4     # thres, tune to prec=recall
n_labels = len( cfg.labels )

gt_roll = []
pred_roll = []
with open( cfg.cv_csv_path, 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)

    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        if fold==curr_fold:
            # get features, tags
            fe_path = fe_fd + '/' + na + '.f'
            info_path = cfg.wav_fd + '/' + na + '.csv'
            tags = ppData.GetTags( info_path )
            y = ppData.TagsToCategory( tags )
            X = cPickle.load( open( fe_path, 'rb' ) )
            
            # aggregate data
            X3d = mat_2d_to_3d( X, agg_num, hop )
    
            p_y_pred = md.predict( X3d )
            p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
            pred = np.zeros(n_labels)
            pred[ np.where(p_y_pred>thres) ] = 1
            pred_roll.append( pred )
            gt_roll.append( y )

pred_roll = np.array( pred_roll )
gt_roll = np.array( gt_roll )

# calculate prec, recall, fvalue
prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
print prec, recall, fvalue