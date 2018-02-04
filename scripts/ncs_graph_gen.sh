MODEL=$1
MODEL_DIR=../bin/$MODEL
MODEL_FILE=$MODEL_DIR/$MODEL.meta
COMPILED_GRAPH=$MODEL_DIR/$MODEL.graph
IN_SZ=224

echo 'Generating Graph For Model for "' $MODEL '"'
mvNCCompile $MODEL_FILE -s 12 -in input -on output -o $COMPILED_GRAPH -is $IN_SZ $IN_SZ

