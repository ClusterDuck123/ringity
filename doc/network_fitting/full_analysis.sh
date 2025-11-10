python generate_ring_network.py --type model_network --nodes 1000 --beta 0.5 --c 1.0 --r 0.3 --gml test_graph.gml --csv test_positions.csv
python retrieve_positions.py --input-filename test_graph.gml --input-true-positions test_positions.csv --output-true-comparison-plot test.pdf

networks=("lipid" "soil" "immune" "fibro" "gene")
for network in ${networks[@]}
do
python fitting_model.py --network-file ../data/empirical_networks/$network.gml --figure-output-folder figures/$network/
done

output_file=data/homophily_scores.csv
networks=("lipid" "soil" "immune" "fibro")
for network in ${networks[@]}
do
    for _ in $(seq 1 1)
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --summary-output-file $output_file
    done
done

networks=("lipid" "soil" "immune" "fibro")
for _ in $(seq 1 100)
do
    for network in ${networks[@]}
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --randomization configuration --summary-output-file $output_file
    done
done

networks=("gene")
for network in ${networks[@]}
do
    for _ in $(seq 1 1)
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --randomization none --summary-output-file $output_file
    done
done

networks=("gene")
for _ in $(seq 1 100)
do
    for network in ${networks[@]}
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --randomization configuration --summary-output-file $output_file
    done
done

python fitting_model.py --network-file ../data/empirical_networks/lipid.gml --display-figures

python draw_violinplots.py data/homophily_scores.csv
