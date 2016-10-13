'''
SUMMARY:  train model on all development set
AUTHOR:   Qiuqiang Kong
Created:  2016.06.28
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data

# hyper-params
agg_num = 11        # concatenate frames
hop = 5            # step_len
act = 'relu'
n_hid = 500
fold = 1
n_out = len( cfg.labels )

# prepare data
tr_X, tr_y, _ = pp_dev_data.GetAllData( cfg.dev_fe_mel_fd, agg_num, hop, fold=None )
[batch_num, n_time, n_freq] = tr_X.shape
print tr_X.shape, tr_y.shape

# build model
seq = Sequential()
seq.add( InputLayer( (n_time, n_freq) ) )
seq.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
seq.add( Dropout( 0.1 ) )
seq.add( Dense( n_hid, act=act ) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( n_hid, act=act ) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( n_hid, act=act ) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( n_out, act='sigmoid' ) )
md = seq.combine()
md.summary()

# validation
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, 
                         metrics=['binary_crossentropy'], call_freq=1, dump_path=None )
                         
# save model
pp_dev_data.CreateFolder( cfg.eva_md_fd )
save_model = SaveModel( dump_fd=cfg.eva_md_fd, call_freq=10 )

# callbacks
callbacks = [ validation, save_model ]

# optimizer
optimizer = Rmsprop(1e-4)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epochs=1000, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 