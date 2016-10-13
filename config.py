'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: 2016.09.13 Add annotation
          2016.10.11 Modify variable name
--------------------------------------
'''

### paths configuration

# development set
dev_root = '/vol/vssp/AP_datasets/audio/dcase2016/chime_home'
dev_wav_fd = dev_root + '/chunks'
dev_anno_fd = dev_wav_fd
dev_cv_csv_path = dev_root + '/development_chunks_refined_crossval_dcase2016.csv'

# evaluation set
eva_root = '/vol/vssp/AP_datasets/audio/dcase2016/task4/chime_home'
eva_wav_fd = eva_root + '/eva_chunks'
eva_anno_fd = eva_root + '/chunk_annotations/annotations'
eva_csv_path = eva_root + '/evaluation_chunks_refined.csv'

# your workspace
scrap_fd = "/vol/vssp/msos/qk/DCASE2016_task4_scrap"

dev_fe_fd = scrap_fd + '/Fe_dev'
dev_fe_mel_fd = dev_fe_fd + '/Mel'
dev_md_fd = scrap_fd + '/Md_dev'
dev_results_fd = scrap_fd + '/Results_dev'

eva_fe_fd = scrap_fd + '/Fe_eva'
eva_fe_mel_fd = eva_fe_fd + '/Mel'
eva_md_fd = scrap_fd + '/Md_eva'
eva_results_fd = scrap_fd + '/Results_eva'



### global configuration

# labels
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.     # sample rate
win = 1024.     # fft window size
n_fold = 5