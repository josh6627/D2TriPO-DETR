#!/usr/bin/env python3
"""Convert MGN-Real to the images/labels/adj/grasps layout used by datasets/vmrd.py.

Requires numpy and Pillow. See scripts/README_metagraspnet.md for conventions.
Exit 2 means annotated samples were withheld; inspect conversion_report.json.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

try:
    from .convert_vmrd import CLASSES
except ImportError:
    from convert_vmrd import CLASSES

NOVEL_ID = 31
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
CLASS_IDS['novel'] = NOVEL_ID
DIRECTORIES = ('images', 'labels', 'adj', 'grasps')


def load_classes(path):
    categories = json.loads(path.read_text())['categories']
    mapping = {}
    for category, entry in categories.items():
        name = entry['vmrd_class']
        if name not in CLASS_IDS:
            raise ValueError(f'Unknown VMRD class {name!r} in {path}')
        mapping[int(category)] = CLASS_IDS[name]
    return categories, mapping


def grasp_rectangle(pair, height):
    """Contact endpoints become opposite jaw-edge midpoints (corners 0-3, 1-2).

    Consequently the loader's center -> midpoint(corner1, corner2) angle follows
    the contact axis, and distance(corner0, corner1) is the jaw opening.
    """
    pair = np.asarray(pair, dtype=float)
    if pair.shape != (2, 2) or not np.isfinite(pair).all():
        raise ValueError('Invalid parallel-jaw contact pair')
    delta = pair[1] - pair[0]
    length = np.linalg.norm(delta)
    if length <= 0 or not math.isfinite(height) or height <= 0:
        raise ValueError('Zero grasp opening or invalid rectangle height')
    offset = np.array([-delta[1], delta[0]]) / length * height / 2
    return np.array([pair[0] - offset, pair[1] - offset,
                     pair[1] + offset, pair[0] + offset])


def convert_order(order, instance_ids):
    """MGN -1 at [below, above] -> VMRD 1 at [above, below].

    Matches upstream Scripts/visualize_layout.py: -order.transpose(). MGN row
    k corresponds to instance ID k+1, including objects invisible in this view.
    """
    order = np.asarray(order)
    if (order.ndim != 2 or order.shape[0] != order.shape[1]
            or not np.isin(order, [-1, 0]).all() or np.any(np.diag(order))):
        raise ValueError('Expected square order matrix containing only 0/-1 and no self edges')
    indices = np.asarray(instance_ids, dtype=int) - 1
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(order):
        raise ValueError('Instance ID outside order matrix')
    # Reject cycles in the full source graph before selecting visible objects.
    graph = -order.T.astype(int)
    remaining = set(range(len(graph)))
    while remaining:
        leaves = {i for i in remaining if not any(graph[i, j] for j in remaining)}
        if not leaves:
            raise ValueError('Cyclic manipulation order')
        remaining -= leaves
    return graph[np.ix_(indices, indices)]


def convert_sample(image_path, mapping, height, difficulty):
    base = image_path.name[:-len('_rgb.png')]
    folder = image_path.parent
    with Image.open(image_path) as image:
        width, image_height = image.size
    with np.load(folder / f'{base}.npz', allow_pickle=False) as archive:
        instances = archive['instances_objects']
        semantic = archive['instances_semantic']
    if instances.shape != (image_height, width) or semantic.shape != instances.shape:
        raise ValueError('Mask and image dimensions disagree')
    if (not np.isfinite(instances).all() or np.any(instances < 0)
            or np.any(instances != np.floor(instances))):
        raise ValueError('Invalid instance mask')
    instance_ids = [int(i) for i in np.unique(instances) if i > 0]
    if not instance_ids:
        raise ValueError('No visible objects')
    labels, objects, classes = [], [], {}
    for instance in instance_ids:
        mask = instances == instance
        categories = np.unique(semantic[mask])
        if len(categories) != 1 or categories[0] not in mapping:
            raise ValueError(f'Instance {instance}: unknown or inconsistent semantic IDs {categories}')
        category = int(categories[0])
        class_id = mapping[category]
        classes[instance] = class_id
        y, x = np.nonzero(mask)
        # Half-open pixel bounds: the maximum occupied pixel is included.
        box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
        labels.append(' '.join(map(str, [class_id, *box, width, image_height])))
        objects.append({'instance_id': instance, 'category_id': category, 'class_id': class_id})
    order = np.loadtxt(folder / f'{base}_order.csv', delimiter=',', ndmin=2)
    adjacency = convert_order(order, instance_ids)
    with np.load(folder / f'{base}_grasps.npz', allow_pickle=False) as archive:
        pairs = archive['parallel_jaw_2d']
    if pairs.size and (pairs.ndim != 3 or pairs.shape[1:] != (2, 2)):
        raise ValueError(f'Invalid parallel_jaw_2d shape {pairs.shape}')
    grasps, dropped = [], []
    for index, pair in enumerate(pairs):
        try:
            corners = grasp_rectangle(pair, height)
            center = np.asarray(pair).mean(axis=0)
            x, y = np.floor(center + 0.5).astype(int)
            if not (0 <= x < width and 0 <= y < image_height):
                raise ValueError('Grasp center outside image')
            instance = int(instances[y, x])
            if instance not in classes:
                raise ValueError('Grasp center is background; instance association unavailable')
            values = [f'{v:.8f}'.rstrip('0').rstrip('.') for v in corners.ravel()]
            grasps.append(' '.join([*values, difficulty, str(instance), str(classes[instance])]))
        except ValueError as exc:
            dropped.append({'grasp_index': index, 'reason': str(exc)})
    if not grasps:
        raise ValueError(f'No assignable parallel-jaw grasps ({len(dropped)} rejected)')
    return {
        'labels': '\n'.join(labels) + '\n',
        'adj': '\n'.join(' '.join(map(str, row)) for row in adjacency) + '\n',
        'grasps': '\n'.join(grasps) + '\n',
    }, {'objects': objects, 'grasp_count': len(grasps), 'dropped_grasps': dropped,
        'invisible_instance_ids': sorted(set(range(1, len(order) + 1)) - set(instance_ids))}


def emit(path, content=None, source=None, check=False):
    """Idempotent writes: never replace different destination content."""
    if os.path.lexists(path):
        if source is not None:
            if not path.is_symlink() or path.resolve() != source.resolve():
                raise ValueError(f'Expected image symlink to {source}: {path}')
        elif path.is_symlink() or not path.is_file() or path.read_text() != content:
            raise ValueError(f'Destination differs: {path}; use a fresh output directory')
    elif check:
        raise ValueError(f'Missing output: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            path.symlink_to(os.path.relpath(source, path.parent))
        else:
            with path.open('x') as handle:
                handle.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/MetaGraspNetRaw'))
    parser.add_argument('--output', type=Path, default=Path('data/metagraspnet'))
    parser.add_argument('--class-map', type=Path,
                        default=Path(__file__).with_name('metagraspnet_classes.json'))
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grasp-height', type=float, default=20,
                        help='Rectangle height in source pixels (MGN only supplies contact pairs)')
    parser.add_argument('--difficulty', choices=('u', 'e', 'h'), default='u',
                        help='u = unavailable (loader maps to 0); MGN has no VMRD e/h labels')
    parser.add_argument('--dry-run', action='store_true', help='Audit without writing')
    parser.add_argument('--check', action='store_true', help='Recompute and verify existing conversion')
    args = parser.parse_args(argv)
    if not 0 <= args.val_fraction <= 1 or not math.isfinite(args.grasp_height) or args.grasp_height <= 0:
        parser.error('val-fraction must be in [0,1] and grasp-height must be finite and positive')
    if args.dry_run and args.check:
        parser.error('--dry-run and --check are mutually exclusive')
    root, output = args.root.resolve(), args.output.resolve()
    if root == output or root in output.parents:
        parser.error('Output must be outside the raw source directory')
    categories, mapping = load_classes(args.class_map)
    images = sorted(root.rglob('*_rgb.png'))
    if not images:
        raise ValueError(f'No *_rgb.png images under {root}')
    scenes = sorted({str(p.parent.relative_to(root)) for p in images})
    random.Random(args.seed).shuffle(scenes)
    count = round(len(scenes) * args.val_fraction)
    if 0 < args.val_fraction < 1 and len(scenes) > 1:
        count = max(1, min(len(scenes) - 1, count))
    validation = set(scenes[:count])
    report = {
        'format_version': 1, 'source_root': str(root), 'class_ids': CLASS_IDS,
        'categories': categories,
        'conventions': {
            'novel': 'ID 31: assumed novel slot after the 31 named VMRD classes',
            'boxes': 'visible instance mask, half-open xyxy pixel bounds',
            'adj': '1 at [above, below]; -source_order.T, selected by instance_id - 1',
            'grasps': 'contact pairs expanded to rectangles; instance from rounded center pixel',
            'grasp_height': args.grasp_height, 'difficulty': args.difficulty,
            'seed': args.seed, 'val_fraction': args.val_fraction,
        },
        'samples': {}, 'skipped': {}, 'blocked': {},
    }
    prepared = []
    totals = {'train': Counter(), 'val': Counter()}
    for number, image in enumerate(images, 1):
        relative = str(image.relative_to(root))
        base = image.name[:-len('_rgb.png')]
        required = [image.parent / f'{base}{suffix}' for suffix in ('.npz', '_grasps.npz', '_order.csv')]
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            report['skipped'][relative] = {'missing': missing}
            continue
        try:
            annotations, metadata = convert_sample(image, mapping, args.grasp_height, args.difficulty)
        except (ValueError, OSError, KeyError) as exc:
            report['blocked'][relative] = str(exc)
            continue
        split = 'val' if str(image.parent.relative_to(root)) in validation else 'train'
        image_id = f'{number:06d}'  # Loader requires an integer filename stem.
        report['samples'][image_id] = {'source': relative, 'split': split, **metadata}
        prepared.append((image_id, split, image, annotations))
        totals[split].update(images=1, objects=len(metadata['objects']), grasps=metadata['grasp_count'],
                             dropped_grasps=len(metadata['dropped_grasps']))
    report['totals'] = {split: dict(counts) for split, counts in totals.items()}
    if not args.dry_run:
        # Preflight existing files and reject stale samples before any writes.
        expected = {(output / directory / split / (image_id + ('.png' if directory == 'images' else '.txt')))
                    for image_id, split, _, _ in prepared for directory in DIRECTORIES}
        for directory in DIRECTORIES:
            for split in ('train', 'val'):
                folder = output / directory / split
                if (output / directory).is_symlink() or folder.is_symlink():
                    raise ValueError(f'Refusing output-directory symlink: {folder}')
                if folder.exists():
                    extras = set(folder.iterdir()) - expected
                    if extras:
                        raise ValueError(f'Unexpected output files in {folder}; use a fresh output directory')
        report_text = json.dumps(report, indent=2) + '\n'
        destinations = [(output / 'conversion_report.json', report_text, None)]
        for image_id, split, image, annotations in prepared:
            destinations.append((output / 'images' / split / f'{image_id}.png', None, image))
            destinations.extend((output / directory / split / f'{image_id}.txt', content, None)
                                for directory, content in annotations.items())
        for path, content, source in destinations:
            if os.path.lexists(path) or args.check:
                emit(path, content, source, check=True)
        if not args.check:
            for directory in DIRECTORIES:
                for split in ('train', 'val'):
                    (output / directory / split).mkdir(parents=True, exist_ok=True)
            for path, content, source in destinations:
                emit(path, content, source)
    for split, counts in totals.items():
        print(f'{split}: {dict(counts)}')
    print(f'Skipped {len(report["skipped"])} views missing annotations; blocked {len(report["blocked"])} samples.')
    print('Dry run; no files written.' if args.dry_run else f'Output verified: {output}')
    return 2 if report['blocked'] or not prepared else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(f'Conversion stopped: {exc}')
