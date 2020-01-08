
TOOLS=../../build/go/bin/tools

mkdir -p ./raw
mkdir -p ./data

echo "Download original data"

wget -O raw/256_ObjectCategories.tar http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar

echo "Ingest"

$TOOLS/caltech256_ingest_canned

echo "Shuffle"

mv data/train_x.dat data/train_x.raw
mv data/train_y.dat data/train_y.raw

$TOOLS/imagenet_shuffle_canned

echo "Done"


