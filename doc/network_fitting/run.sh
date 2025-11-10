source .venv/bin/activate
python doc/network_fitting/main.py --network lipid --model none --make_figs true --output_folder figures/network_fitting/lipid/
python doc/network_fitting/main.py --network fibro --model none --make_figs true --output_folder figures/network_fitting/fibro/
python doc/network_fitting/main.py --network immune --model none --make_figs true --output_folder figures/network_fitting/immune/
python doc/network_fitting/main.py --network soil  --model none --make_figs true --output_folder figures/network_fitting/soil/
python doc/network_fitting/main.py --network gene_corrected  --model none --make_figs true --output_folder figures/network_fitting/gene_corrected/

python doc/network_fitting/draw_layout.py --network lipid --model none --make_figs true --output_folder figures/network_fitting/lipid/
python doc/network_fitting/draw_layout.py --network fibro --model none --make_figs true --output_folder figures/network_fitting/fibro/
python doc/network_fitting/draw_layout.py --network immune --model none --make_figs true --output_folder figures/network_fitting/immune/
python doc/network_fitting/draw_layout.py --network soil  --model none --make_figs true --output_folder figures/network_fitting/soil/
python doc/network_fitting/draw_layout.py --network gene_corrected  --model none --make_figs true --output_folder figures/network_fitting/gene_corrected/
