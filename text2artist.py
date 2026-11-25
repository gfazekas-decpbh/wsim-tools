"""Find WASABI artists for a free-text description or sound prompt.

Usage:  python text2artist.py "dark moody electronic music with a sad female singer" [-k 10] [--concepts]
"""
import argparse
from wsim_text import TextToArtist
ap = argparse.ArgumentParser(); ap.add_argument('text', nargs='+'); ap.add_argument('-k', type=int, default=10)
ap.add_argument('--concepts', action='store_true', help='also show the vocabulary concepts the text was mapped to')
a = ap.parse_args(); t2a = TextToArtist()
for text in a.text:
    res, cons = t2a.query(text, k=a.k, return_concepts=True)
    print(f'\n> {text}')
    if a.concepts: print('  concepts:', ', '.join(cons))
    for r in res.itertuples(): print(f'  {r.score:6.2f}  {r.name}')
