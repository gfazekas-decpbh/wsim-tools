"""wsim_text: map free text to WASABI artists, and artists to tag prompts (wsim-tools).

Text -> artists = Stage A (phrases -> vocabulary concepts -> v0.3 KG-transformer encoder) fused with
Stage B (frozen MiniLM sentence embedding -> trained projection into the v0.3 artist space), plus a small popularity prior.
Requires: torch, numpy, pandas, scipy, sentence-transformers (all-MiniLM-L6-v2 is downloaded on first use).
"""
import json, os, re, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import sparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'encoder'))
from e3_model import ArtistEncoder

STOP = set('a an the and or of with in on at to for from by as is are was were be been being it its this that these those very '
           'really some any music song songs sound sounds sounding track tracks like style vibe feel feeling kind type made make '
           'over under into out up down about without more less much lots full plus band group singer artist'.split())
LIM = {1: 24, 2: 4, 3: 4}; OFF = {1: 0, 2: 24, 3: 28}; LMAX = 32

def _zrow(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)

class TextToArtist:
    def __init__(self, here=HERE):
        from sentence_transformers import SentenceTransformer
        self.cfg = json.load(open(f'{here}/text-model/config.json')); v = json.load(open(f'{here}/encoder/vocab.json'))
        self.concepts = v['concepts']; self.ctype = np.array([{'tag': 1, 'genre': 2, 'subject': 3}[c.split(':', 1)[0]] for c in self.concepts])
        self.cname = [c.split(':', 1)[1] for c in self.concepts]
        order = sorted(range(len(self.cname)), key=lambda i: -{1: 1, 2: 3, 3: 0}[self.ctype[i]])
        self.exact = {}
        for i in order: self.exact.setdefault(self.cname[i], i)
        self.st = SentenceTransformer(self.cfg['sentence_model'], device='cpu')
        self.C = np.load(f'{here}/text-model/concept_embeddings_minilm.npy').astype(np.float32)
        self.enc = ArtistEncoder(len(self.concepts) + 1, v['d'], use_bias=v['use_bias'])
        self.enc.load_state_dict(torch.load(f'{here}/encoder/encoder.pt')); self.enc.eval()
        E = np.load(f'{here}/encoder/concept_graph_edges.npy')
        self.A = sparse.csr_matrix((E[:, 2], (E[:, 0].astype(int), E[:, 1].astype(int))), shape=(len(self.concepts) + 1,) * 2)
        b = self.cfg['stageB']; self.head = (nn.Linear(384, 128) if b['head'] == 'linear' else
                                             nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 128)))
        self.head.load_state_dict({k.removeprefix('net.'): v for k, v in torch.load(f'{here}/text-model/projection.pt').items()}); self.head.eval()
        art = pd.read_parquet(f'{here}/artist_embeddings_v0.3.parquet'); self.art = art
        self.Z = np.stack(art.vector.values).astype(np.float32); self.pop = art.popularity_z.values.astype(np.float32)

    def _phrases(self, text):
        w = re.findall(r"[a-z0-9&'+-]+", text.lower().replace('’', "'")); out = []
        for n in (1, 2, 3):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if g[0] not in STOP and g[-1] not in STOP: out.append(' '.join(g))
        return list(dict.fromkeys(out))

    def concepts_for(self, text):
        a = self.cfg['stageA']; ph = self._phrases(text); W = {}
        for p in ph:
            if p in self.exact: W[self.exact[p]] = 1.0
        q = [p for p in ph if len(p) > 2] + ([text] if a['sentence'] else [])
        if q:
            S = self.st.encode(q, normalize_embeddings=True) @ self.C.T
            for r in range(len(q)):
                k = 8 if (a['sentence'] and r == len(q) - 1) else int(a['topk'])
                for j in np.argpartition(-S[r], k)[:k]:
                    if S[r, j] >= a['tau']: W[int(j)] = max(W.get(int(j), 0), float(S[r, j]))
        return W

    def _encode(self, W):
        seq = torch.zeros(1, LMAX, dtype=torch.long); ty = torch.zeros(1, LMAX, dtype=torch.long); wt = torch.zeros(1, LMAX); kg = torch.zeros(1, LMAX, LMAX)
        for t in (1, 2, 3):
            items = sorted([(w, j) for j, w in W.items() if self.ctype[j] == t], reverse=True)[:LIM[t]]
            tot = sum(w for w, _ in items)
            for r, (w, j) in enumerate(items): seq[0, OFF[t] + r] = j + 1; ty[0, OFF[t] + r] = t; wt[0, OFF[t] + r] = w / tot
        if not (seq > 0).any(): return np.zeros(128, np.float32)
        nz = seq[0].nonzero().flatten().numpy(); toks = seq[0, nz].numpy(); kg[0][np.ix_(nz, nz)] = torch.from_numpy(self.A[toks][:, toks].toarray())
        with torch.no_grad(): return self.enc(seq, ty, wt, kg)[0].numpy()

    def query(self, text, k=10, return_concepts=False):
        f = self.cfg['fusion']; W = self.concepts_for(text)
        sA = _zrow((self._encode(W) @ self.Z.T)[None])[0]
        with torch.no_grad():
            e = torch.from_numpy(self.st.encode([text], normalize_embeddings=True)); zb = F.normalize(self.head(e), dim=-1)[0].numpy()
        sB = _zrow((zb @ self.Z.T)[None])[0]
        S = (sB if f['beta'] == 'B' else sA + float(f['beta']) * sB) + float(f['pop']) * self.pop
        top = np.argsort(-S)[:k]
        res = self.art.iloc[top][['artist', 'name', 'mbid']].assign(score=np.round(S[top], 3)).reset_index(drop=True)
        if return_concepts:
            return res, [self.concepts[j] for j, _ in sorted(W.items(), key=lambda x: -x[1])[:15]]
        return res

def artist_prompt(name_or_uri, by_facet=False, here=HERE):
    """The artist's TAG prompt: genres + most characteristic descriptive Last.fm tags (listener tags, not generated text).
    Default order = the artist's own tag weight; by_facet=True orders the same tags by facet (genre/style, instruments,
    vocals, mood, tempo, era, place, other)."""
    art = pd.read_parquet(f'{here}/artist_tag_prompts_v0.3.2.parquet')
    m = art[(art.artist == name_or_uri) | (art.name.str.lower() == str(name_or_uri).lower())]
    return None if m.empty else m.iloc[0]['prompt_by_facet' if by_facet else 'prompt']
