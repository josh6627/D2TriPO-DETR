#!/usr/bin/env python3
"""Copy a reproducible percentage of converted VMRD train and validation data.

Uses only the Python standard library. Source files are never modified.
"""
import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile


SPLITS = ('train', 'val')
FOLDERS = ('images', 'labels', 'adj', 'grasps')


def percentage(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError('percentage must be a number in (0, 100]') from exc
    if not result.is_finite() or not 0 < result <= 100:
        raise argparse.ArgumentTypeError('percentage must be in (0, 100]')
    return result


def plan_subset(root, percent, seed):
    """Validate file bundles and sample within each existing split independently."""
    selected, counts = {}, {}
    for split in SPLITS:
        folder = root / 'images' / split
        if not folder.is_dir():
            raise ValueError(f'Missing converted image directory: {folder}. Convert VMRD first.')
        images = sorted(p for p in folder.iterdir() if p.suffix in ('.jpg', '.jpeg', '.png'))
        if not images:
            raise ValueError(f'No images in {folder}')
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f'Duplicate image stems in {folder}')
        count = int((len(images) * percent / 100).to_integral_value(rounding=ROUND_HALF_UP))
        if count == 0:
            raise ValueError(f'{percent}% rounds to zero {split} images; choose a larger percentage.')
        # A separate local generator avoids changing global random state.
        chosen = sorted(random.Random(f'{seed}:{split}').sample(images, count))
        for image in chosen:
            for directory in FOLDERS:
                path = image if directory == 'images' else root / directory / split / (image.stem + '.txt')
                if not path.is_file():
                    raise ValueError(f'Missing selected sample file: {path}')
        selected[split] = chosen
        counts[split] = {'available': len(images), 'selected': count}
    # VMRD IDs identify images globally; never move an ID into another split.
    all_ids = {}
    for split in SPLITS:
        all_ids[split] = {p.stem for p in (root / 'images' / split).iterdir()
                          if p.suffix in ('.jpg', '.jpeg', '.png')}
    overlap = all_ids['train'] & all_ids['val']
    if overlap:
        raise ValueError(f'Train/validation ID overlap: {sorted(overlap)[:10]}')
    return selected, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/vmrd'), help='Converted VMRD root')
    parser.add_argument('--output', required=True, type=Path, help='New dataset directory')
    parser.add_argument('--percent', required=True, type=percentage, help='Percentage of EACH split (e.g. 5)')
    parser.add_argument('--seed', type=int, default=42, help='Repeatable random selection (default: 42)')
    parser.add_argument('--dry-run', action='store_true', help='Preview selection without creating files')
    args = parser.parse_args(argv)
    root, output = args.input.resolve(), args.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Input and output must not contain one another')
    if os.path.lexists(args.output):
        raise ValueError(f'Output already exists: {args.output}; choose a new directory')

    selected, counts = plan_subset(root, args.percent, args.seed)
    # Keep numeric class IDs unchanged, including classes absent from this subset.
    classes_path = root / 'classes.json'
    if classes_path.is_file():
        classes = json.loads(classes_path.read_text())
    else:
        report_path = root / 'conversion_report.json'
        if not report_path.is_file():
            raise ValueError(f'Missing classes.json or conversion_report.json in {root}')
        mapping = json.loads(report_path.read_text())['class_ids']
        if (not isinstance(mapping, dict) or any(type(i) is not int for i in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError(f'Invalid class IDs in {report_path}')
        classes = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name.strip() for name in classes)
            or len(classes) != len(set(classes))):
        raise ValueError('Class registry must contain unique, nonempty class names')
    source_manifest = root / 'manifest.json'
    metadata = json.loads(source_manifest.read_text()) if source_manifest.is_file() else {}
    report = {
        'source_root': str(root), 'percent': float(args.percent), 'seed': args.seed,
        'counts': counts,
        'selected_ids': {split: [p.stem for p in paths] for split, paths in selected.items()},
    }
    for split, count in counts.items():
        print(f"{split}: {count['selected']}/{count['available']} images "
              f"({100 * count['selected'] / count['available']:.2f}%)")
    print(f"Train:val ratio: {counts['train']['available'] / counts['val']['available']:.4f}:1 -> "
          f"{counts['train']['selected'] / counts['val']['selected']:.4f}:1 (whole-image rounding)")
    if args.dry_run:
        print('Dry run: no files written.')
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage complete copies alongside the destination, then publish the directory.
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as staging:
        staged = Path(staging) / 'dataset'
        samples = {}
        for split, images in selected.items():
            for directory in FOLDERS:
                (staged / directory / split).mkdir(parents=True, exist_ok=True)
            for image in images:
                for directory in FOLDERS:
                    source = image if directory == 'images' else root / directory / split / (image.stem + '.txt')
                    # Dereference converted VMRD's image symlinks for a portable subset.
                    shutil.copy2(source, staged / directory / split / source.name)
                samples[image.stem] = {
                    **metadata.get('samples', {}).get(image.stem, {}),
                    'split': split, 'source_image': str(image),
                }
        manifest = {'tasks': metadata.get('tasks', {'grasps': True}), 'samples': samples}
        for filename, content in (('classes.json', classes), ('manifest.json', manifest),
                                  ('subset_report.json', report)):
            (staged / filename).write_text(json.dumps(content, indent=2) + '\n')
        if os.path.lexists(output):
            raise ValueError(f'Output already exists: {output}')
        staged.rename(output)
    print(f'Created {output}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
