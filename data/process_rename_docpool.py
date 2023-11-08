import hashlib
import os

import docsaidkit as D

DIR = D.get_curdir(__file__)

fs = D.get_files(DIR / 'docpool', suffix=['.jpg', '.png', '.jpeg', '.webp'])

for f in D.Tqdm(fs):
    img = D.imread(f)
    image_bytes = img.tobytes()
    md5_hash = hashlib.md5(image_bytes).hexdigest()

    # rename
    new_name = f'{md5_hash}.jpg'

    os.rename(str(f), str(DIR / 'docpool' / new_name))
