
source .venv/bin/activate

mkdir data/
mkdir data/parameter_array/

beta_values=("0.8" "0.85" "0.9" "0.95" "1.0")
r_values=("0.1" "0.15" "0.2" "0.25")

for beta in "${beta_values[@]}";
do

    for r in "${r_values[@]}";
    do

        for i in $(seq 1 1)
        do
           echo iteration $1 of "python make_network.py --output_folder data/parameter_array/ --n_nodes 200 --r $r --beta $beta --c 1.0"
           python make_network.py --output_folder data/parameter_array/ --n_nodes 200 --r $r --beta $beta --c 1.0
        done;

    done;


done
