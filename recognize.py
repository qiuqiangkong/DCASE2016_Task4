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
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
from Hat.metrics import prec_recall_fvalue
import Hat.backend as K
import config as cfg
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 50
hop = 50
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md/md95.p', 'rb' ) )

# prepare data
tr_Xlist, tr_ylist, te_Xlist, te_ylist = ppData.GetListData( fe_fd, agg_num, hop, fold=1 )

# do recognize and evaluation
thres = 0.5     # thres, tune to prec=recall
n_labels = len( cfg.labels )

pred_roll = []  # shape: (n_audio, n_label)
gt_roll = np.array( te_ylist )  # shape: (n_audio, n_label)
for X in te_Xlist:
    p_y_preds = md.predict( X )     # shape:(n_block, n_label)
    p_y_pred = np.mean( p_y_preds, axis=0 )     # shape:(n_label)
    pred = np.zeros(n_labels)
    pred[ np.where(p_y_pred>thres) ] = 1
    pred_roll.append( pred )

pred_roll = np.array( pred_roll )

# calculate prec, recall, fvalue
prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
print prec, recall, fvalue