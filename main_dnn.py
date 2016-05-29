'''
SUMMARY:  Dcase 2016 Task 4. Audio Tagging
          Training time: 1 s/epoch. (Tesla M2090)
          test f_value: 73% (thres=0.5), test EER=24%  after 50 epoches     
          Try adjusting hyper-params, optimizer, longer epoches to get better results. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.29
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
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )

# prepare data
tr_Xlist, tr_ylist, te_Xlist, te_ylist = ppData.GetListData( fe_fd, agg_num, hop, fold=1 )
tr_X, tr_y = ppData.ListToMat( tr_Xlist, tr_ylist )
te_X, te_y = ppData.ListToMat( te_Xlist, te_ylist )
[batch_num, n_time, n_freq] = tr_X.shape

print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape

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
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, 
                         metric_types=['binary_crossentropy', 'prec_recall_fvalue'], call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
# optimizer = SGD( 0.01, 0.95 )
optimizer = Rmsprop(0.001)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=101, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 