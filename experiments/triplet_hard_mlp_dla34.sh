cd src
python train.py mot --exp_id triplet_hard_mlp --unsup --unsup_loss triplet_hard --mlp_layer --reid_dim 1024 --gpus 0 --batch_size 2 --load_model '/home/danielzgsilva/Documents/SelfSupMOT/src/lib/models/ctdet_coco_dla_2x.pth'
cd ..
