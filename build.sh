#!/bin/bash

usage() { echo "Usage: $0 -m <modeldir>"; exit 1; }

while getopts ":m:l:" o; do
    case "${o}" in
        m)
            m=${OPTARG};;
        *)
            usage
            ;;
    esac
done

echo "\n--------------- Freezing graph -----------------------------\n";
freeze_graph --input_saved_model_dir=${m} --output_graph=${m}/frozen.pb --output_node_names=final_result;

echo "\n--------------- Optimizing graph ---------------------------\n";
python -m tensorflow.python.tools.optimize_for_inference --input=${m}/frozen.pb --output=${m}/optimized.pb --input_names="Placeholder" --output_names="final_result"

echo "\n--------------- Adding graph to Tensorboard ----------------\n"
python -m scripts.graph_pb2tb ${m} ${m}/optimized.pb;

echo "\n--------------- Generating quantized graphs ----------------\n"
#python -m scripts.quantize_graph --input=${m}/optimized.pb --output=${m}/round.pb --output_node_names=final_result --mode=round;

#python -m scripts.quantize_graph --input=${m}/optimized.pb --output=${m}/quantize.pb --output_node_names=final_result --mode=quantize;

python -m scripts.quantize_graph --input=${m}/optimized.pb --output=${m}/eightbit.pb --output_node_names=final_result --mode=eightbit;

python -m scripts.quantize_graph --input=${m}/optimized.pb --output=${m}/weights.pb --output_node_names=final_result --mode=weights;

python -m scripts.quantize_graph --input=${m}/optimized.pb --output=${m}/weights_rounded.pb --output_node_names=final_result --mode=weights_rounded;

echo "\n--------------- DONE ---------------------------------------\n"
echo "Directory path: ${m}"
