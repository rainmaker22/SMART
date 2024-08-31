mkdir pyg_depend && cd pyg_depend
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_cluster-1.6.0%2Bpt112cu102-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_scatter-2.1.0%2Bpt112cu102-cp38-cp38-linux_x86_64.whl
python3 -m pip install torch_cluster-1.6.0+pt112cu102-cp38-cp38-linux_x86_64.whl
python3 -m pip install torch_scatter-2.1.0+pt112cu102-cp38-cp38-linux_x86_64.whl
python3 -m pip install torch_geometric
