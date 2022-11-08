export LD_LIBRARY_PATH=/opt/zlib/1.2.9/lib:/usr/lib64:$LD_LIBRARY_PATH

# Download local cocodataset
python3 trivial/prep.py

# trivial query benchmarks 
python3 trivial/query.py --scenario=0
python3 trivial/query.py --scenario=1
python3 trivial/query.py --scenario=2

# trivial query benchmarks 
python3 trivial/query.py --stream --scenario=0
python3 trivial/query.py --stream --scenario=1
python3 trivial/query.py --stream --scenario=2

# python dataloader
python3 trivial/dataloader.py
python3 trivial/dataloader.py --shuffle
python3 trivial/dataloader.py --shuffle --model

# python stream dataloader
python3 trivial/dataloader.py --stream
python3 trivial/dataloader.py --stream --shuffle
python3 trivial/dataloader.py --stream --shuffle --model

# cpp dataloader
python3 trivial/dataloader.py --enterprise
python3 trivial/dataloader.py --enterprise --shuffle 
python3 trivial/dataloader.py --enterprise --shuffle --model 

# cpp stream dataloader
# python3 trivial/dataloader.py --enterprise --stream
# python3 trivial/dataloader.py --enterprise --stream --shuffle 
# python3 trivial/dataloader.py --enterprise --stream --shuffle --model 
