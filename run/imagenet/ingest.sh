
TOOLS=../../build/go/bin/tools

# ./raw must exist and contain TAR files for all involved synsets

mkdir -p ./data

echo "Download synset list"

$TOOLS/imagenet_list_2012

echo "Ingest"

$TOOLS/imagenet_ingest_canned

echo "Shuffle"

mv data/train_x.dat data/train_x.raw
mv data/train_y.dat data/train_y.raw

$TOOLS/imagenet_shuffle_canned

echo "Done"


