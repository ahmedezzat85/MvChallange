MODEL=$1
MODEL_DIR=../bin/$MODEL
MODEL_FILE=$MODEL_DIR/$MODEL.meta
COMPILED_GRAPH=compiled.graph
IN_SZ=$2

echo 'Generating Graph For Model for "' $MODEL  $IN_SZ'"'
mvNCCompile $MODEL_FILE -s 12 -in input -on output -o $COMPILED_GRAPH -is $IN_SZ $IN_SZ

