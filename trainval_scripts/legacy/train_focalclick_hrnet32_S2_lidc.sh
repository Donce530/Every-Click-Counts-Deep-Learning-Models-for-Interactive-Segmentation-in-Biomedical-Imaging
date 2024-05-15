CUDA_VISIBLE_DEVICES=2,3
python train.py models/focalclick/hrnet32_lidc_3c.py\
  --gpus=2,3\
  --workers=16\
  --batch-size=32\
  --exp-name=hrnet32_S2_lidc\
