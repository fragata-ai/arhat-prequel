
mkdir -p bin/tools

go build -o bin/tools/caltech256_ingest_canned fragata/arhat/tools/caltech256/ingest_canned

go build -o bin/tools/imagenet_list_2012 fragata/arhat/tools/imagenet/list_2012
go build -o bin/tools/imagenet_gather_synset fragata/arhat/tools/imagenet/gather_synset
go build -o bin/tools/imagenet_ingest_canned fragata/arhat/tools/imagenet/ingest_canned
go build -o bin/tools/imagenet_shuffle_canned fragata/arhat/tools/imagenet/shuffle_canned


