cd src
python train.py mot --exp_id only_self_sup_dla34 --unsup --gpus 0 --batch_size 2 --load_model '/home/danielzgsilva/Documents/SelfSupMOT/src/lib/models/ctdet_coco_dla_2x.pth'
cd ..
