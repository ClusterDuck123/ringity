source .venv/bin/activate
python notebooks/network_fitting/main.py --network lipid --model none --make_figs true --output_folder figures/lipid/
python notebooks/network_fitting/main.py --network fibro --model none --make_figs true --output_folder figures/fibro/
python notebooks/network_fitting/main.py --network immune --model none --make_figs true --output_folder figures/immune/
python notebooks/network_fitting/main.py --network soil  --model none --make_figs true --output_folder figures/soil/


for i in {1..10}
do
    python notebooks/network_fitting/main.py --network soil --model configuration --make_figs false --output_folder chopchop/
done

python notebooks/network_fitting/main.py --network gene_corrected  --model none --make_figs true --output_folder figures/genes/
