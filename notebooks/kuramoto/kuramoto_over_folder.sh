


source .venv/bin/activate

N_RUNS_PER_NETWORK=1



for i in $(seq 1 $N_RUNS_PER_NETWORK);
do
   for network_folder in $(ls data/parameter_array/);
    do echo iteration $i of  "python run_kuramoto.py --i data/parameter_array/$network_folder --summary-info-file data/parameter_array_summary.csv --only-summary True"
    python run_kuramoto.py --i data/parameter_array/$network_folder --summary-info-file data/parameter_array_summary.csv --only-summary True;
    done;
done
