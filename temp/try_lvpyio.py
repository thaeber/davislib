# %%
from pathlib import Path

import lvpyio as lv


#%%
class ImageSet(collections.abc.Sized)

# %%
path = Path(r'D:\MyProjects\First NO LIF\Scan-002-250124-165124\WL=226.020')
lv.is_multiset(path)

# %%
s = lv.read_set(path)

# %%
s
# %%
