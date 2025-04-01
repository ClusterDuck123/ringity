
source .venv/bin_activate
python make_networks.py test/
bash load_network_run_kuramoto.sbatch
python parameter_array_dotplot.py --i test/ --o test_dotplot.png --terminal_length 10 --threshold 0.001
