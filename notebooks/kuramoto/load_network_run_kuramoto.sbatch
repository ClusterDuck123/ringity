#SBATCH --array=0-1000
#SBATCH --mem=5G 
#SBATCH --time=01:00:00 

source .venv/bin/activate
network_folder=/lisc/project/test
python load_network_run_kuramoto.py --i $network_folder