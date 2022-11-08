import argparse
import deeplake

parser = argparse.ArgumentParser(description="python dataloader")

parser.add_argument("--stream", action='store_true', default=False)
parser.add_argument("--scenario", type=int, default=0, metavar="O", help="number of workers to allocate")

if __name__ == '__main__':
    args = parser.parse_args()
    print("dataloader: ", args)
    
    if args.stream:
        ds = deeplake.load('hub://activeloop/coco-val')
    else:
        ds = deeplake.load('./dataset/coco-val')
    
    if args.scenario == 0: 
        ds_view1 = ds.query('select * where not contains(categories, \'truck\')')

    elif args.scenario == 1:
        ds_view2 = ds.query('(select * where contains(categories, \'car\') limit 1000) \
                union (select * where contains(categories, \'motorcycle\') \
                limit 1000)')

    elif args.scenario == 2:
        ds_view3 = ds.query('select * where any(logical_and(boxes[:,3]>500, categories == \'truck\'))')

