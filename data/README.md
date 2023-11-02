# Dataset

- MIDV-500/MIDV-2019:

  URL: https://github.com/fcakyon/midv500

  1. Install midv500 package:

      ```bash
      pip install mdiv500
      ```

  2. Run `python download_midv.py` to download the dataset.

- MIT Indoor Scenes:

  https://web.mit.edu/torralba/www/indoor.html

- CORD v0:

  https://github.com/clovaai/cord

# Build dataset

Make sure you have downloaded the dataset and put them in the right place.

Setting the `ROOT` in `build_dataset.py` to the root directory of the dataset.

And then run:

```bash
python build_dataset.py
```
