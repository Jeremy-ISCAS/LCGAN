# LCGAN
Key code of unsupervised learning method LCGAN


Train models:
python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path /data/self-learning-data/Ships \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 96 \
--epoch_queue_starts 15




//Evaluate models: Linear classification on ImageNet//
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
--data_path 
--pretrained /home/detection/sly/LCGAN/multiview-wdgrl/checkpoints/ckp-250.pth
