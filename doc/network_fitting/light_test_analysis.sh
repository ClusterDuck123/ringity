source .venv/bin/activate

mkdir data/
mkdir data/light_test/
mkdir data/light_test/figures/
mkdir data/light_test/figures/control_homophily_violin_plots/

python generate_ring_network.py --type model_network --nodes 100 --beta 0.5 --c 1.0 --r 0.3 --gml data/light_test/test_graph.gml --csv data/light_test/test_positions.csv
python retrieve_positions.py --input-filename data/light_test/test_graph.gml --input-true-positions data/light_test/test_positions.csv --output-true-comparison-plot data/light_test/figures/test.pdf

networks=("lipid" "soil" "immune" "fibro")
for network in ${networks[@]}
do
python fitting_model.py --network-file ../data/empirical_networks/$network.gml --figure-output-folder data/light_test/figures/$network/
done

output_file=data/light_test/homophily_scores.csv
networks=("lipid" "soil" "immune" "fibro")
for network in ${networks[@]}
do
    for _ in $(seq 1 1)
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --summary-output-file $output_file
    done
done

networks=("lipid" "soil" "immune" "fibro")
for _ in $(seq 1 10)
do
    for network in ${networks[@]}
    do
    python fitting_model.py --network-file ../data/empirical_networks/$network.gml --randomization configuration --summary-output-file $output_file
    done
done

python draw_violinplots.py $output_file data/light_test/figures/control_homophily_violin_plots/

#python fitting_model.py --network-file ../data/empirical_networks/lipid.gml --display-figures True
