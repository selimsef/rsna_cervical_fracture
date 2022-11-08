DATA_DIR=$1
FOLD=$2


echo "Training ${CONFIG} fold ${FOLD} "
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4  --master_port 9971 \
  train_cls.py --world-size 4 --distributed --fold $FOLD --config configs/clsr152fr.json --data-dir $DATA_DIR \
  --test_every 1 --fp16 --workers 10 --prefix full_ --output-dir weights/ \
  --resume weights/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20200803-fc66ce8d.pth --from-zero