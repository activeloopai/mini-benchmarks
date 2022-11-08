import argparse

import deeplake
from deeplake.experimental import dataloader

from utils import transform_train, collate_fn, get_model_object_detection, train_one_epoch, train, iterate 

parser = argparse.ArgumentParser(description="python dataloader")

parser.add_argument("--stream", action='store_true', default=False)
parser.add_argument("--shuffle", action='store_true', default=False)
parser.add_argument("--model", action='store_true', default=False)
parser.add_argument("--enterprise", action='store_true', default=False)

if __name__ == '__main__':
    args = parser.parse_args()
    print("dataloader: ", args)

    if args.stream:
        ds = deeplake.load('hub://activeloop/coco-val')
    else:
        ds = deeplake.load('./dataset/coco-val')

    if args.model:
        ds = ds[:32]

    if args.enterprise:
        train_loader = dataloader(ds)\
                    .transform(transform_train)\
                    .batch(32)\
                    .pytorch(tensors = ['images', 'categories', 'boxes'], num_workers = 8, collate_fn=collate_fn)\
                    .shuffle(args.shuffle)
    else:
        train_loader = ds.pytorch(
                          num_workers = 8, 
                          shuffle = args.shuffle, 
                          transform = transform_train,
                          tensors = ['images', 'categories', 'boxes'],
                          batch_size = 32,
                          collate_fn = collate_fn
                        )
    if args.model:
        train(train_loader)
    else:
        iterate(train_loader)

    print('Finished')