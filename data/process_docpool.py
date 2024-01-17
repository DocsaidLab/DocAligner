import docsaidkit as D

DIR = D.get_curdir(__file__)

fs = D.get_files(DIR / 'doc_type_dataset', suffix=['.jpg', '.jpeg', '.png'])

ROOT = DIR / 'unique_pool'

ROOT_PRIVATE = ROOT / 'private'

if not ROOT.is_dir():
    ROOT.mkdir(parents=True)

if not ROOT_PRIVATE.is_dir():
    ROOT_PRIVATE.mkdir(parents=True)

for f in D.Tqdm(fs):
    img = D.imread(f)
    name = D.img_to_md5(img)
    if 'private' in str(f):
        D.imwrite(img, ROOT_PRIVATE / f'{name}.jpg')
    else:
        D.imwrite(img, ROOT / f'{name}.jpg')
