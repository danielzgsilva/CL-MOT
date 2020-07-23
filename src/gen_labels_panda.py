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
    print(seq)
    seq_info = json.load(open(osp.join(seq_root, seq, 'seqinfo.json')))
    seq_width = int(seq_info['imWidth'])
    seq_height = int(seq_info['imHeight'])

    tracks = json.load(open(osp.join(seq_root, seq, 'tracks.json')))

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    print(len(tracks))
    print(tracks[0].keys())

    print(seq_width, seq_height)

    '''for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)'''
