
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 9896 --use_env main.py --backbone resnet50 --ds_name sgg_vg --lab_name detr_300-no_img_10-obj_att-bbox --batch_size 6 --epochs 10
