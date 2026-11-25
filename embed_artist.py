"""Embed any artist from a list of tags/genres/subjects with the v0.3 encoder (works for artists not in WASABI).

Usage:  python embed_artist.py --tags "shoegaze,dream pop,noise pop" --genres "alternative rock" [--neighbours 10]
Needs: encoder/ (encoder.pt, e3_model.py, vocab.json, concept_graph_edges.npy) and, for --neighbours,
       artist_embeddings_v0.3.parquet in the same folder as this script.
"""
import argparse, json, os, sys, numpy as np, torch
from scipy import sparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'encoder'))
from e3_model import ArtistEncoder
ap = argparse.ArgumentParser(); ap.add_argument('--tags', default=''); ap.add_argument('--genres', default=''); ap.add_argument('--subjects', default='')
ap.add_argument('--neighbours', type=int, default=10); args = ap.parse_args()
V = json.load(open(os.path.join(HERE, 'encoder/vocab.json'))); cid = {c: i + 1 for i, c in enumerate(V['concepts'])}
E = np.load(os.path.join(HERE, 'encoder/concept_graph_edges.npy'))
A = sparse.csr_matrix((E[:, 2], (E[:, 0].astype(int), E[:, 1].astype(int))), shape=(len(cid) + 1,) * 2)
seq, typ, unknown = [], [], []
for kind, txt, lim, t in [('tag', args.tags, V['max_tags'], 1), ('genre', args.genres, V['max_genres'], 2), ('subject', args.subjects, V['max_subjects'], 3)]:
    toks = [x.strip().lower() for x in txt.split(',') if x.strip()][:lim]
    for x in toks:
        if f'{kind}:{x}' in cid: seq.append(cid[f'{kind}:{x}']); typ.append(t)
        else: unknown.append(f'{kind}:{x}')
if unknown: print('ignored (not in vocabulary):', ', '.join(unknown))
if not seq: sys.exit('no known tokens')
L = V['max_tags'] + V['max_genres'] + V['max_subjects']
s = torch.zeros(1, L, dtype=torch.long); ty = torch.zeros(1, L, dtype=torch.long); w = torch.zeros(1, L)
s[0, :len(seq)] = torch.tensor(seq); ty[0, :len(seq)] = torch.tensor(typ); w[0, :len(seq)] = 1 / len(seq)
kg = torch.zeros(1, L, L); kg[0, :len(seq), :len(seq)] = torch.from_numpy(A[seq][:, seq].toarray())
m = ArtistEncoder(len(cid) + 1, V['d'], use_bias=V['use_bias']); m.load_state_dict(torch.load(os.path.join(HERE, 'encoder/encoder.pt'))); m.eval()
with torch.no_grad(): z = m(s, ty, w, kg)[0].numpy()
print('embedding (first 8 dims):', np.round(z[:8], 3))
emb_path = os.path.join(HERE, 'artist_embeddings_v0.3.parquet')
if args.neighbours and os.path.exists(emb_path):
    import pandas as pd
    df = pd.read_parquet(emb_path); Z = np.stack(df.vector.values); sc = Z @ z
    for i in np.argsort(-sc)[:args.neighbours]: print(f'{sc[i]:.3f}  {df.name.iloc[i]}')
