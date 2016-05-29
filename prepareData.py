'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
import numpy as np
from scipy import signal
import cPickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scikits.audiolab import wavread
import librosa
import config as cfg
import csv
import scipy.stats

# Use preemphasis, the same as matlab
def GetMel( wav_fd, fe_fd ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        if na.endswith('.48kHz.wav'):
            wav, fs, enc = wavread( path )
            if ( wav.ndim==2 ): 
                wav = np.mean( wav, axis=-1 )
            assert fs==48000
            ham_win = np.hamming(1024)
            [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
            X = X.T
            
            if globals().get('melW') is None:
                global melW
                melW = librosa.filters.mel( fs, n_fft=1024, n_mels=40, fmin=0., fmax=22100 )
                melW /= np.max(melW, axis=-1)[:,None]
                
            X = np.dot( X, melW.T )
            
            #print X.shape
            #plt.matshow(X.T, origin='lower', aspect='auto')
            #plt.show()
            #pause
            
            out_path = fe_fd + '/' + na[0:-10] + '.f'
            cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
          
# get tags
def GetTags( info_path ):
    with open( info_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y
            
    
# value in list is 3d (n_block, n_time, n_freq) audio features
def GetListData( fe_fd, agg_num, hop, fold ):
    with open( cfg.cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        [len_X, n_freq] = X.shape
        X3d = []    # shape: (n_block, n_time, n_freq)
        i1 = 0
        while ( i1+agg_num<len_X ):
            X3d.append( X[i1:i1+agg_num] )
            i1 += hop
        X3d = np.array( X3d )
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist.append( y )
        else:
            tr_Xlist.append( X3d )
            tr_ylist.append( y )
        
    return tr_Xlist, tr_ylist, te_Xlist, te_ylist
    
# concatenate list of (n_block, n_time, n_freq) to (batch_num, n_time, n_freq)
def ListToMat( Xlist, ylist ):
    Xall = np.concatenate( Xlist, axis=0 )
    yall = []
    for i1 in xrange( len(Xlist) ):
        len_X = Xlist[i1].shape[0]
        for i2 in xrange( len_X ):
            yall.append( ylist[i1] )
    yall = np.array( yall )
    return Xall, yall
            
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
            
if __name__ == "__main__":
    CreateFolder('Fe')
    CreateFolder('Fe/Mel')
    CreateFolder('Results')
    CreateFolder('Md')
    GetMel( cfg.wav_fd, cfg.fe_mel_fd )