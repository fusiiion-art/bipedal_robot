#!/usr/bin/env python3
import pickle
from pathlib import Path

p = pickle.load(open(Path('log/version_7/final_params.pkl'), 'rb'))
print('top_type', type(p))
if isinstance(p, tuple):
    print('tuple_len', len(p))
    for i, e in enumerate(p):
        print('idx', i, 'type', type(e))
        if hasattr(e, 'keys'):
            try:
                keys = list(e.keys())
                print(' keys', keys[:20])
            except Exception as ex:
                print(' keys_error', ex)


def walk(tree, prefix=''):
    if isinstance(tree, dict):
        for k, v in tree.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if 'std' in str(k).lower() or 'scale' in str(k).lower():
                print('match_key', p, 'type', type(v))
            walk(v, p)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            walk(v, f"{prefix}[{i}]")


for idx in (1, 2):
    if idx < len(p) and isinstance(p[idx], dict) and 'params' in p[idx]:
        print('--- walk tuple index', idx, '---')
        walk(p[idx]['params'], f'tuple[{idx}]/params')
