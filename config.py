wav_fd = '/homes/qkong/datasets/DCASE2016/Domestic audio tagging/chime_home/chunks'
fe_mel_fd = 'Fe/Mel'

cv_csv_path = '/homes/qkong/datasets/DCASE2016/Domestic audio tagging/chime_home/development_chunks_refined_crossval_dcase2016.csv'

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }
