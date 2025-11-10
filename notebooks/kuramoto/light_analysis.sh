
source .venv/bin/activate

mkdir data/light_test
mkdir data/light_test/parameter_array/
mkdir data/light_test/search_for_diverse_dynamics/


beta_values=("0.8" "0.85" "0.9" "0.95" "1.0")
r_values=("0.1" "0.15" "0.2" "0.25")

for beta in "${beta_values[@]}";
do

    for r in "${r_values[@]}";
    do

        for i in $(seq 1 1)
        do
           echo iteration $i of "python make_network.py --output_folder data/light_test/parameter_array/ --n_nodes 200 --r $r --beta $beta --c 1.0"
           python make_network.py --output_folder data/light_test/parameter_array/ --n_nodes 200 --r $r --beta $beta --c 1.0
        done;

    done;

done

N_RUNS_PER_NETWORK=1

for i in $(seq 1 $N_RUNS_PER_NETWORK);
do
   for network_folder in $(ls data/light_test/parameter_array/);
    do echo iteration $i of  "python run_kuramoto.py --i data/parameter_array/$network_folder --summary-info-file data/light_test/parameter_array_summary.csv --only-summary True"
    python run_kuramoto.py --i data/light_test/parameter_array/$network_folder --summary-info-file data/light_test/parameter_array_summary.csv --only-summary True;
    done;
done

python parameter_array_dotplot.py  --i data/light_test/parameter_array_summary.csv --o figures/light_test/dotplot.svg --threshold default 0.00000001
python ring_score_coherence_scatterplot.py --i data/light_test/parameter_array_summary.csv --o figures/light_test/scatterplot.svg

beta_values=("0.7" "1.0")
r_values=("0.1" "0.25")

for beta in "${beta_values[@]}";
do

    for r in "${r_values[@]}";
    do

        for i in $(seq 1 1)
        do
           echo iteration $1 of "python make_network.py --output_folder data/search_for_diverse_dynamics/ --n_nodes 200 --r $r --beta $beta --c 1.0"
           python make_network.py --output_folder data/light_test/search_for_diverse_dynamics/ --n_nodes 200 --r $r --beta $beta --c 1.0
        done;

    done;
done

N_RUNS_PER_NETWORK=1

for i in $(seq 1 $N_RUNS_PER_NETWORK);
do
   for network_folder in $(ls data/light_test/search_for_diverse_dynamics/);
    do echo iteraction $i of  "python run_kuramoto.py --i data/light_test/parameter_array/$network_folder --summary-info-file data/light_test/search_for_diverse_dynamics.csv"
    python run_kuramoto.py --i data/light_test/search_for_diverse_dynamics/$network_folder --summary-info-file data/light_test/search_for_diverse_dynamics.csv;
    done;
done

python select_and_plot_coherence.py --target_number 3 --input-csv data/light_test/search_for_diverse_dynamics.csv --input-folder data/light_test/parameter_array/ --output-folder figures/reruns/
