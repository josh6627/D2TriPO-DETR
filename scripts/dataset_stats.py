#!/usr/bin/env python3
"""Count annotated object appearances and directed relationships without a model.

Uses only the Python standard library. Input annotations are never modified.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys


SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def load_classes(root, override=None):
    path = override or root / 'classes.json'
    if override or path.is_file():
        names = json.loads(path.read_text())
    else:
        path = root / 'conversion_report.json'
        if not path.is_file():
            return None
        names = json.loads(path.read_text()).get('class_ids')
    if isinstance(names, dict):
        if (any(type(i) is not int for i in names.values())
                or set(names.values()) != set(range(len(names)))):
            raise ValueError(f'{path}: class IDs must be unique and contiguous starting at zero')
        names = [name for name, _ in sorted(names.items(), key=lambda item: item[1])]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)):
        raise ValueError(f'{path}: expected a nonempty class list or name-to-ID map')
    if len({' '.join(n.split()).casefold() for n in names}) != len(names):
        raise ValueError(f'{path}: duplicate class names')
    return names


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def count_sample(label_path, adj_path, names):
    counts = Counter()
    labels = read_rows(label_path)
    for number, row in enumerate(labels, 1):
        if len(row) != 7:
            raise ValueError(f'{label_path}: row {number}: expected class xmin ymin xmax ymax width height')
        try:
            class_id = int(row[0])
            x1, y1, x2, y2, width, height = map(float, row[1:])
        except ValueError as exc:
            raise ValueError(f'{label_path}: row {number}: invalid numeric field') from exc
        if class_id < 0 or (names is not None and class_id >= len(names)):
            raise ValueError(f'{label_path}: row {number}: unknown/invalid class ID {class_id}')
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, width, height))
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError(f'{label_path}: row {number}: invalid box or dimensions')
        counts[class_id] += 1
    adj = read_rows(adj_path)
    n = len(labels)
    if len(adj) != n or any(len(row) != n for row in adj):
        raise ValueError(f'{adj_path}: expected {n} x {n} adjacency matrix')
    if any(value not in ('0', '1') for row in adj for value in row):
        raise ValueError(f'{adj_path}: adjacency must contain only 0 and 1')
    if any(adj[i][i] != '0' for i in range(n)):
        raise ValueError(f'{adj_path}: adjacency diagonal must be zero')
    return counts, sum(value == '1' for row in adj for value in row)


def collect_stats(root, splits, names):
    results, warnings, errors = {}, [], []
    for split in splits:
        folder = root / 'images' / split
        if not folder.is_dir():
            raise ValueError(f'Missing image directory: {folder}')
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        stems = Counter(p.stem for p in images)
        duplicates = sorted(stem for stem, count in stems.items() if count > 1)
        if duplicates:
            raise ValueError(f'{folder}: duplicate image stems: {duplicates}')
        for kind in ('labels', 'adj'):
            annotation_dir = root / kind / split
            if not annotation_dir.is_dir():
                raise ValueError(f'Missing annotation directory: {annotation_dir}')
            orphans = sorted(p.name for p in annotation_dir.glob('*.txt') if p.stem not in stems)
            if orphans:
                warnings.append(f'{annotation_dir}: ignored {len(orphans)} orphan annotation(s): '
                                + ', '.join(orphans[:10]) + (' ...' if len(orphans) > 10 else ''))
        counts, edges = Counter(), 0
        for image in images:
            try:
                sample_counts, sample_edges = count_sample(
                    root / 'labels' / split / (image.stem + '.txt'),
                    root / 'adj' / split / (image.stem + '.txt'), names)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                continue
            counts.update(sample_counts)
            edges += sample_edges
        results[split] = {'images': len(images), 'counts': counts, 'relationships': edges}
    if errors:
        raise ValueError('No totals produced because annotation validation failed:\n' + '\n'.join(errors))
    combined = {'images': 0, 'counts': Counter(), 'relationships': 0}
    for result in results.values():
        combined['images'] += result['images']
        combined['counts'].update(result['counts'])
        combined['relationships'] += result['relationships']
    results['combined'] = combined
    return results, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('data/vmrd'))
    parser.add_argument('--split', choices=('all', *SPLITS), default='all',
                        help='Default: all existing train/val/test image directories')
    parser.add_argument('--classes', type=Path, help='Override registry with a class list or name-to-ID map')
    parser.add_argument('--csv', type=Path, help='Write per-class instance counts by split to this CSV')
    args = parser.parse_args(argv)
    root = args.dataset_root
    names = load_classes(root, args.classes)
    splits = ([s for s in SPLITS if (root / 'images' / s).is_dir()]
              if args.split == 'all' else [args.split])
    if not splits:
        raise ValueError(f'{root}: no train/val/test image directories found')
    results, warnings = collect_stats(root, splits, names)
    for warning in warnings:
        print(f'Warning: {warning}', file=sys.stderr)
    if names is None:
        print('Warning: no class registry found; reporting numeric IDs only. '
              'Use --classes to supply names.', file=sys.stderr)
    print(f'Dataset: {root.absolute()}')
    print('Instances count appearances across images; relationships count saved directed edges.')
    print(f'{"Split":<12} {"Images":>10} {"Objects":>12} {"Classes present":>16} {"Relationships":>15}')
    for split, result in results.items():
        print(f'{split:<12} {result["images"]:>10} {sum(result["counts"].values()):>12} '
              f'{len(result["counts"]):>16} {result["relationships"]:>15}')
    ids = list(range(len(names))) if names is not None else sorted(results['combined']['counts'])
    if names is not None:
        absent = len(names) - len(results['combined']['counts'])
        print(f'Registered classes: {len(names)}; classes with zero instances: {absent}')
    rows = [{'class_id': i, 'class_name': names[i] if names is not None else f'class_{i}',
             **{split: result['counts'][i] for split, result in results.items()}} for i in ids]
    print('\nInstances per class:')
    name_width = max([10] + [len(row['class_name']) for row in rows])
    print(f'{"ID":>4}  {"Class":<{name_width}}' + ''.join(f'{split:>12}' for split in results))
    for row in rows:
        print(f'{row["class_id"]:>4}  {row["class_name"]:<{name_width}}'
              + ''.join(f'{row[split]:>12}' for split in results))
    if args.csv:
        # Exclusive creation prevents accidentally overwriting annotations or earlier reports.
        with args.csv.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['class_id', 'class_name', *results])
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nPer-class CSV: {args.csv}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
