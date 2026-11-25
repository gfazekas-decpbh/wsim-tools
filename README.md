# wsim-tools: artist embeddings and text tools for WASABI

Artist embeddings, trained models and small command-line tools that work with the [WASABI artist similarity graphs](https://doi.org/10.5281/zenodo.24511585). There are tools to:
- find real artists from a free-text description;
- turn an artist into a keyword tag prompt;
- embed a new artist from tags;
- run nearest-neighbour search over 36,839 WASABI artists.

By George Fazekas, Centre for Digital Music, Queen Mary University of London.

## Install
```bash
pip install -r requirements.txt      # torch, numpy, pandas, scipy, pyarrow, sentence-transformers
```
The text tools download the MiniLM sentence model (`sentence-transformers/all-MiniLM-L6-v2`, about 90 MB) from Hugging Face on first use.

## Tools
| Command | What it does |
|---|---|
| `python text2artist.py "dreamy shoegaze with whispered vocals" -k 10 [--concepts]` | Free text → closest real artists. `--concepts` shows which tags the text was mapped to |
| `python artist2prompt.py "Portishead" "Daft Punk" [--by-facet]` | Artist → **tag prompt**: genres plus descriptive Last.fm tags (listener tags, no names, not generated text). `--by-facet` orders them by facet |
| `python embed_artist.py --tags "trip-hop,downtempo,female vocalists" --genres "trip hop" --neighbours 5` | Embed a **new** artist from exact vocabulary tags and genres |

From Python:
```python
from wsim_text import TextToArtist, artist_prompt
t2a = TextToArtist()                                   # about 5 s to load; about 0.15 s per query on a CPU
t2a.query("80s synthpop with drum machines and catchy hooks", k=5)   # DataFrame: artist, name, mbid, score
#   Orchestral Manoeuvres In The Dark, Yazoo, Erasure, The Human League, Soft Cell
artist_prompt("Miles Davis")                 # 'trumpet, cool jazz, bebop, instrumental, jazz trumpet, pure jazz, 50s, hard bop, …'
artist_prompt("Miles Davis", by_facet=True)  # 'cool jazz, bebop, instrumental, pure jazz, hard bop, bop, experimental, trumpet, …'

import numpy as np, pandas as pd                       # plain nearest neighbours on the vectors
E = pd.read_parquet('artist_embeddings_v0.3.parquet'); Z = np.stack(E.vector)
i = E.index[E.name == 'Radiohead'][0]; E.name.values[np.argsort(-(Z @ Z[i]))[1:6]]
#   Talk Talk, Thom Yorke, DEUS, Talking Heads, Peter Gabriel
```

## Files
| File | Content |
|---|---|
| `artist_embeddings_v0.3.parquet` | One row per artist: `artist` (WASABI URI), `name`, `mbid`, `vector` (128-d, unit length, so cosine = dot product), `popularity_z` |
| `artist_embeddings_v0.3_sample.csv` | Five artists from that table, readable on GitHub (vector as space-separated numbers) |
| `artist_tag_prompts_v0.3.2.parquet` | Per-artist tag prompts used by `artist2prompt.py`: `prompt` (the artist's own tag order) and `prompt_by_facet` |
| `encoder/encoder.pt`, `e3_model.py` | Transformer artist encoder: weights and model definition |
| `encoder/vocab.json`, `concept_graph_edges.npy` | Concept vocabulary (32.5k Last.fm tags, genres and Wikipedia categories) and the concept graph |
| `text-model/projection.pt`, `config.json` | Trained projection from MiniLM sentence embeddings into the artist space, and all tuned settings |
| `text-model/concept_embeddings_minilm.npy` | MiniLM embeddings of the 32.5k concept names |
| `wsim_text.py`, `text2artist.py`, `artist2prompt.py`, `embed_artist.py` | Module and command-line tools |

Artists are identified by WASABI URIs (`http://ns.inria.fr/wasabi/artist/…`), the same identifiers as in the similarity graphs and in WASABI itself.

## How they were built
- **Artist vectors and encoder.** A two-layer transformer reads each artist's Last.fm tags, genres and Wikipedia categories from the WASABI knowledge graph. Its token embeddings start from ComplEx knowledge-graph embeddings and are kept fixed. The rest is trained on MTG-SAS similar-artist pairs. These vectors are one of the three components of the cultural similarity graph (v0.3).
- **Text → artists.** Two routes are combined:
  - the text is mapped to known tags with MiniLM, then passed through the encoder;
  - a small projection, trained on templated tag descriptions and masked artist biographies, maps MiniLM sentence embeddings directly into the artist space.

  On 40 hand-written sound prompts, 74% of the top-10 artists carry an expected label.
- **Tag prompts.** Each prompt is the artist's genres plus their most characteristic descriptive Last.fm tags. User names, artist names, meta tags and offensive tags are removed. Run back through the search, a prompt returns artists of the original artist's genre 70% of the time.

## Related
- **Similarity graphs** (cultural, chord-profile, within-genre dissimilarity, chord theory, tag prompts), as Turtle, JSON-LD and CSV: Zenodo [10.5281/zenodo.24511585](https://doi.org/10.5281/zenodo.24511585). The Turtle and JSON-LD files are also in the [WASABI fork](https://github.com/gfazekas-decpbh/WasabiDataset) (`similarity/`).
- **Vocabulary:** `wsim:` = `https://wsim.semanticaudio.net/ontology/wsim-0.1.ttl#`.
- **Source data:** the WASABI Song Corpus, [10.5281/zenodo.5603369](https://doi.org/10.5281/zenodo.5603369).

## Tips
- **Vague queries.** Queries with a few broad tags (e.g. `embed_artist.py` with three tags) tend to return obscure artists whose whole profile is just those tags. `text2artist.py` adds a mild popularity prior; set `fusion.pop` in `text-model/config.json` to 0 for discovery, or raise it for well-known names.
- **Vocabulary terms.** `embed_artist.py` needs exact vocabulary terms (see `encoder/vocab.json`, e.g. `tag:trip-hop`, `genre:trip hop`); unknown terms are skipped with a warning. Use `text2artist.py` for free text.
- **Text-to-audio.** Tag prompts are keyword lists, which suits text-to-audio models. Try `--by-facet` if you want sound-related tags first.
- **Coverage.** Coverage follows WASABI and Last.fm: popular Western genres are well covered, and some scenes barely appear.

## Licence
- **Data and models** (embeddings, tag prompts, encoder, text model, vocabulary): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), following the licence of the WASABI data they are derived from.
- **Code** (`*.py`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

See [`LICENSE`](LICENSE) for which files fall under each.

