import os
import json

panda_root = 'PANDA/images/train/'
data_root = os.path.join(os.getcwd(), 'data')
seq_root = os.path.join(data_root, 'MOT', 'PANDA', 'images', 'train')

seqs = [s for s in os.listdir(seq_root)]

output_file = 'panda.train'

for seq in seqs:
    fnames = os.listdir(os.path.join(seq_root, seq))
    seq_info = json.load(open(os.path.join(seq_root, seq, 'seqinfo.json')))
    img_ext = seq_info['imExt']

    for fname in fnames:
        file_str = os.path.join(panda_root, seq, fname)

        if fname.endswith(img_ext):
            print(file_str)
