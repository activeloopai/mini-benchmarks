# Intel Benchmarks

Has two set of benchmarks, one for dataloader and query.

### Setup

```
pip3 install -r requirements.txt
python3 trivial/prep.py
```

### Query

an example query

```
python3 trivial/query.py --stream --scenario=0
```

### Dataloader

an example dataloader
```
python3 trivial/dataloader.py --enterprise --stream --shuffle --model 
```


### All run

```
./trivial.sh
```
