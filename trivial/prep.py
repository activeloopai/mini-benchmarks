import deeplake

if __name__ == '__main__':
    ds = deeplake.load('hub://activeloop/coco-val')

    if not deeplake.exists('./dataset/coco-val'):
        deeplake.deepcopy('hub://activeloop/coco-val', './dataset/coco-val', num_workers=8, scheduler='processed', progressbar=True)