'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: 
--------------------------------------
'''
root = '/homes/qkong/datasets/DCASE2016/Domestic audio tagging/chime_home'
wav_fd = root + '/chunks'

# development
fe_mel_fd = 'Fe/Mel'
cv_csv_path = root + '/development_chunks_refined_crossval_dcase2016.csv'

# evaluation
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.
win = 1024.