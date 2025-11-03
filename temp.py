from genrec.utils import get_config
from genrec.datasets.AmazonReviews2023.dataset import AmazonReviews2023

cfg = get_config('ActionPiece', 'AmazonReviews2023', None, {'category': 'CDs_and_Vinyl'})
ds = AmazonReviews2023(cfg)

print('n_items:', ds.n_items)
print('n_users:', ds.n_users)
print('n_interactions:', ds.n_interactions)
print('metadata entries:', 0 if ds.item2meta is None else len(ds.item2meta))