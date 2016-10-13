'''
SUMMARY:  Dcase 2016 Task 4. Audio Tagging
          Training time: 1 s/epoch. (Tesla M2090) 
          Try adjusting hyper-params, optimizer, longer epoches to get better results. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.29
Modified: 2016.10.11 modify variable name
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
from sklearn import preprocessing
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import Adam
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data


# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 1            # step_len
act = 'relu'
n_hid = 500
fold = 0
n_out = len( cfg.labels )

def train():
    # prepare data
    tr_X, tr_y, _, te_X, te_y, te_na_list = pp_dev_data.GetAllData( fe_fd, agg_num, hop, fold )
    [batch_num, n_time, n_freq] = tr_X.shape
    
    print tr_X.shape, tr_y.shape
    print te_X.shape, te_y.shape
    
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
    
    # optimizer
    optimizer = Adam(1e-4)
    
    # callbacks
    # tr_err, te_err are frame based. To get event based err, run recognize.py
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=2000, 
                            metrics=['binary_crossentropy'], call_freq=1, dump_path=None )
                            
    # save model
    pp_dev_data.CreateFolder( cfg.dev_md_fd )
    save_model = SaveModel( dump_fd=cfg.dev_md_fd, call_freq=10 )
    
    # callbacks
    callbacks = [ validation, save_model ]
    
    # fit model
    md.fit( x=tr_X, y=tr_y, batch_size=2000, n_epochs=100, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks, verbose=1 )
    
    
if __name__ == '__main__':
    train()