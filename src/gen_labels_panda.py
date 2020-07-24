import os.path as osp
import os
import numpy as np
import json

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = os.path.join(os.getcwd(), 'data', 'MOT', 'PANDA', 'images', 'train')
label_root = os.path.join(os.getcwd(), 'data', 'MOT', 'PANDA', 'labels_with_ids', 'train')
mkdirs(label_root)

seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = json.load(open(osp.join(seq_root, seq, 'seqinfo.json')))

    img_fnames = seq_info['imUrls']
    img_ext = seq_info['imExt']
    seq_width = int(seq_info['imWidth'])
    seq_height = int(seq_info['imHeight'])

    tracks = json.load(open(osp.join(seq_root, seq, 'tracks.json')))

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for track in tracks:
        tid = int(track['track id'])
        frames = track['frames']

        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid

        for frame in frames:
            fid = int(frame['frame id'])
            img_fname = img_fnames[fid - 1]

            bbox = frame['rect']

            tlbr = np.array([bbox['tl']['x'], bbox['tl']['y'], bbox['br']['x'], bbox['br']['y']], dtype=np.float64)

            # height and width of bbox
            w = bbox['br']['x'] - bbox['tl']['x']
            h = bbox['br']['y'] - bbox['tl']['y']

            # x, y center of bbox
            x = (bbox['br']['x'] + bbox['tl']['x']) / 2
            y = (bbox['br']['y'] + bbox['tl']['y']) / 2

            label_fpath = osp.join(seq_label_root, img_fname.replace(img_ext, '.txt'))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid_curr, x, y, w, h)

            with open(label_fpath, 'a') as f:
                f.write(label_str)
