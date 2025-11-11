source .venv/bin/activate

mkdir data/
mkdir data/search_for_diverse_dynamics/

beta_values=("0.7" "1.0")
r_values=("0.1" "0.25")

for beta in "${beta_values[@]}";
do

    for r in "${r_values[@]}";
    do

        for i in $(seq 1 10)
        do
           echo iteration $1 of "python make_network.py --output_folder data/search_for_diverse_dynamics/ --n_nodes 200 --r $r --beta $beta --c 1.0"
           python make_network.py --output_folder data/search_for_diverse_dynamics/ --n_nodes 200 --r $r --beta $beta --c 1.0
        done;

    done;
done

N_RUNS_PER_NETWORK=10

for i in $(seq 1 $N_RUNS_PER_NETWORK);
do
   for network_folder in $(ls data/search_for_diverse_dynamics/);
    do echo iteraction $i of  "python run_kuramoto.py --i data/parameter_array/$network_folder --summary-info-file data/search_for_diverse_dynamics.csv"
    python run_kuramoto.py --i data/search_for_diverse_dynamics/$network_folder --summary-info-file data/search_for_diverse_dynamics.csv;
    done;
done
