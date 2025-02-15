
srun -p digitalcity -N1 --quotatype=reserved --gres=gpu:4 --cpus-per-task=64 --job-name=coco

COCO_DIR=/mnt/petrelfs/yangmengping/data/coco
mkdir -p $COCO_DIR

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $COCO_DIR/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -O $COCO_DIR/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip -O $COCO_DIR/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $COCO_DIR/val2017.zip

unzip $COCO_DIR/annotations_trainval2017.zip -d $COCO_DIR
unzip $COCO_DIR/stuff_annotations_trainval2017.zip -d $COCO_DIR
unzip $COCO_DIR/train2017.zip -d $COCO_DIR/images
unzip $COCO_DIR/val2017.zip -d $COCO_DIR/images
