'''
SUMMARY:  train model on all development set
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
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import SGD, Rmsprop
import Hat.backend as K
import config as cfg
import prepareData_eva as ppData_eva

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
fold = 1
n_out = len( cfg.labels )

# prepare data
tr_X, tr_y = ppData_eva.GetAllData( fe_fd, cfg.cv_csv_path, agg_num, hop )
[batch_num, n_time, n_freq] = tr_X.shape
print tr_X.shape, tr_y.shape

# build model
md = Sequential()
md.add( InputLayer( (n_time, n_freq) ) )
md.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_out, act='sigmoid' ) )
md.summary()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, 
                         metric_types=['binary_crossentropy', 'prec_recall_fvalue'], call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md_eva', call_freq=10 )
callbacks = [ validation, save_model ]

# optimizer
# optimizer = SGD( 0.01, 0.95 )
optimizer = Rmsprop(1e-4)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epoch=1001, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 