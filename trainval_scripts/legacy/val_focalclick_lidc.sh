python scripts/evaluate_model.py FocalClick\
  --model_dir=/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/experiments/focalclick/lidc_hrnet32/008_hrnet_32-3-channel-lidc-focal/checkpoints\
  --checkpoint=last_checkpoint\
  --infer-size=256\
  --datasets=LIDC_2D_VAL\
  --gpus=0\
  --n-clicks=20\
  --target-iou=0.95\
  --thresh=0.5\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\