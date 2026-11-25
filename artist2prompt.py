"""Artist -> TAG prompt: the artist's genres and most characteristic descriptive Last.fm tags (listener tags, no names),
usable as a keyword prompt for text-to-audio models.

Usage:  python artist2prompt.py "Portishead" "Daft Punk" [--by-facet]
"""
import sys
from wsim_text import artist_prompt
by_facet = '--by-facet' in sys.argv
for name in [a for a in sys.argv[1:] if a != '--by-facet']:
    p = artist_prompt(name, by_facet=by_facet); print(f'{name}: {p if p else "(not found, or fewer than 2 descriptive tags)"}')
