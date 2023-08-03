try:
    import midv500
except ImportError:
    print('Please install the midv500 library with: pip install midv500')

dataset_dir = 'midv500_data/'
dataset_name = "all"
midv500.download_dataset(dataset_dir, dataset_name)
