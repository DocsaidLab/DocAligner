import hashlib
import os

import docsaidkit as D

DIR = D.get_curdir(__file__)

fs = D.get_files(DIR / 'docpool', suffix=['.jpg', '.png', '.jpeg', '.webp'])

for f in D.Tqdm(fs):
    new_name = D.gen_md5(f) + '.jpg'
    os.rename(str(f), str(DIR / 'docpool1' / new_name))
