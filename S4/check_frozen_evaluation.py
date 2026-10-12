"""Offline checks for frozen evidence, atomic checkpoints and model-named exports."""
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


base = Path(__file__).resolve().parent
notebook = json.loads((base / 'S4_01_hypothesis_evaluation.ipynb').read_text())
namespace = dict(Path=Path, hashlib=hashlib, json=json, re=re, math=math,
                 csv=csv, subprocess=subprocess, display_path=str)
names = {'sha256', 'model_slug', 'model_file', 'save_checkpoint', 'verify_frozen_corpus'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    compile(source, f'cell_{index}', 'exec')
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'notebook_functions', 'exec'), namespace)

with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)
    corpus = folder / 'literature_corpus.json'
    corpus.write_text(json.dumps({'query_version': 'published_works_v3',
                                 'records': [{'hypothesis_id': 'H1', 'papers': []}]}))
    expected = namespace['sha256'](corpus)
    verify = namespace['verify_frozen_corpus']
    verify(folder, ['H1'], 'stage3_hash', expected)
    verify(folder, ['H1'], 'stage3_hash', expected)  # unchanged restart succeeds
    for ids, stage3_hash in [(['H2'], 'stage3_hash'), (['H1'], 'changed')]:
        try:
            verify(folder, ids, stage3_hash, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed provenance was accepted')
    corpus.write_text(corpus.read_text() + '\n')
    try:
        verify(folder, ['H1'], 'stage3_hash', expected)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed corpus was accepted')
    checkpoint = namespace['model_file'](folder, 'deepseek-r1:14b', 'evaluation_records.json')
    namespace['save_checkpoint'](checkpoint, [{'candidate_ids': ['H1']}])
    namespace['save_checkpoint'](checkpoint, [{'candidate_ids': ['H1']}, {'candidate_ids': ['H2']}])
    assert len(json.loads(checkpoint.read_text())) == 2
    assert not checkpoint.with_suffix('.json.tmp').exists()

    original = base / 'outputs/S4_01_hypothesis_evaluation_final_20260929_001548_130904'
    record = json.loads((original / 'qwen3_30b/qwen3_30b_evaluation_records.json').read_text())[0]
    namespace['DIMENSIONS'] = json.loads((original / 'rubric_source_map.json').read_text())['dimensions']
    namespace['representatives'] = [record]
    exec(''.join(notebook['cells'][8]['source']), namespace)
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.instance()  # Reproduce the scripted notebook runner's non-GUI shell.
    namespace['export_model_outputs']('qwen3:30b', folder, [record])
    for name in ['literature_grounded_hypothesis_priority.csv', 'expert_review_template.csv',
                 'hypothesis_evaluation.docx', 'hypothesis_spider_charts.pdf', 'METHOD_AND_RESULTS.md']:
        assert (folder / f'qwen3_30b_{name}').stat().st_size > 0, name
    assert (folder / 'qwen3_30b_hypothesis_spider_png/qwen3_30b_review_01.png').is_file()
print('PASS: frozen evidence/provenance, restart checks, atomic checkpoint and named exports.')
