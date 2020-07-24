import os

panda_root = 'PANDA/images/train/'
data_root = os.path.join(os.getcwd(), 'data')
seq_root = os.path.join(data_root, 'MOT', 'PANDA', 'images', 'train')
print(seq_root)
seqs = [s for s in os.listdir(seq_root)]

output_file = 'panda.train'

for seq in seqs:
    print(seq)
    fnames = os.listdir(os.path.join(seq_root, seq))

    print(fnames)
