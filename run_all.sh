DATA_DIR=$1
echo "Download ig65/kinetics pretrained weights"
sh download_weights.sh

echo "Preprocess images"
PYTHONPATH=. python -u datatools/preprocess.py --root_dir $DATA_DIR

echo "Run Segmentation training"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4  --master_port 9971 \
 train_unet_3d.py --world-size 4 --distributed --fold 1 --config configs/r50csn3d.json --data-dir $DATA_DIR --test_every 1 \
 --fp16 --workers 10 --prefix 256_ --output-dir weights/ \
 --resume weights/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth --from-zero

echo "Predict Segmentation Masks"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4  --master_port 9971 \
  predict_segmentation.py --world-size 4 --distributed  --config r50csn3d --weights-path weights \
  --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds --checkpoint 256_ResNet3dCSN2P1D_r50ir_0_dice

echo "Compute metadata"
PYTHONPATH=. python -u compute_meta.py --root_dir $DATA_DIR

echo "Training ir-r152CSN folds"

sh train_cls_ir.sh $DATA_DIR 0
sh train_cls_ir.sh $DATA_DIR 1
sh train_cls_ir.sh $DATA_DIR 2
sh train_cls_ir.sh $DATA_DIR 3



echo "Training ip-r152CSN folds"

sh train_cls_ip.sh $DATA_DIR 0
sh train_cls_ip.sh $DATA_DIR 1
sh train_cls_ip.sh $DATA_DIR 2
sh train_cls_ip.sh $DATA_DIR 3

python average_weights.py --path weights/ --exp_name full_ClassifierResNet3dCSN2P1D_r152ir_0 --num_best 5 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_ClassifierResNet3dCSN2P1D_r152ir_1 --num_best 5 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_ClassifierResNet3dCSN2P1D_r152ir_2 --num_best 5 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_ClassifierResNet3dCSN2P1D_r152ir_3 --num_best 5 --metric_name loss --min_epoch 0

python average_weights.py --path weights/ --exp_name full_frozen_ClassifierResNet3dCSN2P1D_r152ip_0 --num_best 3 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_frozen_ClassifierResNet3dCSN2P1D_r152ip_1 --num_best 3 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_frozen_ClassifierResNet3dCSN2P1D_r152ip_2 --num_best 3 --metric_name loss --min_epoch 0
python average_weights.py --path weights/ --exp_name full_frozen_ClassifierResNet3dCSN2P1D_r152ip_3 --num_best 3 --metric_name loss --min_epoch 0