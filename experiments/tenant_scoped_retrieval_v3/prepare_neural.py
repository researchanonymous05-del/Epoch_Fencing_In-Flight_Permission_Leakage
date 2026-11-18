"""Encode saved synthetic source records offline using the cached model."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
records=json.loads((ROOT/'data/benchmark.json').read_text())['records']
allowed=['kind','title','statement','catalog_kind','skus','eligible_skus','eligible_plans','segment','lines','amount','currency','credit_amount','asserted_net','line_prices']
texts=[' '.join(f'{k} {json.dumps(r[k],sort_keys=True)}' for k in allowed if k in r) for r in records]
m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
v=m.encode(texts,batch_size=16,normalize_embeddings=True,show_progress_bar=False).astype('float32')
import numpy as np
np.save(H/'data/neural_vectors.npy',v)
print('Offline neural vectors:',v.shape)
