#!/usr/bin/env python3
"""Merge exported D2TriPO-DETR datasets by class name, preserving train/val/test splits.

Requires Pillow, but no model, GPU, or desktop. Run --dry-run to validate inputs.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

try:
    from .convert_vmrd import CLASSES
except ImportError:
    from convert_vmrd import CLASSES

SPLITS = ('train', 'val', 'test')
FOLDERS = ('images', 'labels', 'adj', 'grasps')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def name_key(name):
    return ' '.join(name.split()).casefold()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def registry(value, path):
    if isinstance(value, dict):
        # VMRD conversion reports store name -> ID rather than a list.
        if (not value or any(not isinstance(n, str) or type(i) is not int for n, i in value.items())
                or set(value.values()) != set(range(len(value)))):
            raise ValueError(f'{path}: class map must have contiguous, unique IDs starting at zero')
        value = [name for name, _ in sorted(value.items(), key=lambda item: item[1])]
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name.strip() for name in value)):
        raise ValueError(f'{path}: expected an ordered list of nonempty class names')
    if len({name_key(name) for name in value}) != len(value):
        raise ValueError(f'{path}: duplicate class names after case/whitespace normalization')
    return [' '.join(name.split()) for name in value]


def source_classes(root, override=None):
    if override:
        return registry(read_json(override), override), str(override)
    path = root / 'classes.json'
    if path.is_file():
        return registry(read_json(path), path), str(path)
    path = root / 'conversion_report.json'
    if path.is_file() and 'class_ids' in (report := read_json(path)):
        return registry(report['class_ids'], path), str(path)
    raise ValueError(f'{root}: no class registry. Supply classes.json or --class-map {root}=PATH')


def tokens(path, required=True):
    if not path.is_file():
        if required:
            raise ValueError(f'Missing annotation: {path}')
        return []
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def remap_id(token, mapping, path):
    try:
        source_id = int(token)
    except ValueError as exc:
        raise ValueError(f'{path}: invalid class ID {token!r}') from exc
    if source_id not in mapping:
        raise ValueError(f'{path}: class ID {source_id} is absent from the source registry')
    return source_id, str(mapping[source_id])


def annotations(root, split, image, mapping, metadata):
    label_path = root / 'labels' / split / (image.stem+'.txt')
    labels = tokens(label_path)
    if not labels:
        raise ValueError(f'{label_path}: at least one object is required')
    with Image.open(image) as source:
        size = source.size
    source_ids = []
    for row in labels:
        if len(row) != 7:
            raise ValueError(f'{label_path}: expected class xmin ymin xmax ymax width height')
        source_id, mapped = remap_id(row[0], mapping, label_path)
        source_ids.append(source_id)
        x1, y1, x2, y2, width, height = map(float, row[1:])
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, width, height))
                or (width, height) != size or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError(f'{label_path}: nonfinite/invalid box or image dimensions')
        row[0] = mapped
    count = len(labels)
    adj_path = root / 'adj' / split / (image.stem+'.txt')
    adjacency = tokens(adj_path)
    if len(adjacency) != count or any(len(row) != count for row in adjacency):
        raise ValueError(f'{adj_path}: expected {count} x {count} adjacency matrix')
    if any(value not in ('0', '1') for row in adjacency for value in row):
        raise ValueError(f'{adj_path}: adjacency entries must be binary')
    if any(adjacency[i][i] != '0' for i in range(count)):
        raise ValueError(f'{adj_path}: diagonal must be zero')
    remaining = set(range(count))
    while remaining:
        leaves = {i for i in remaining if not any(adjacency[i][j] == '1' for j in remaining)}
        if not leaves:
            raise ValueError(f'{adj_path}: cyclic relationships')
        remaining -= leaves
    owners = metadata.get('object_ids_in_row_order')
    if owners is not None and (not isinstance(owners, list) or len(owners) != count
                               or any(type(i) is not int or i <= 0 for i in owners)
                               or len(set(owners)) != count):
        raise ValueError(f'{root}/manifest.json: invalid object IDs for {image.name}')
    owner_classes = dict(zip(owners, source_ids)) if owners is not None else None
    grasp_path = root / 'grasps' / split / (image.stem+'.txt')
    grasps = tokens(grasp_path, required=False)
    for row in grasps:
        if len(row) != 11:
            raise ValueError(f'{grasp_path}: expected 8 coordinates, difficulty, instance ID, class ID')
        source_id, mapped = remap_id(row[-1], mapping, grasp_path)
        values = list(map(float, row[:8]))
        owner = int(row[-2])
        if (not all(math.isfinite(v) for v in values) or owner <= 0
                or row[-3] not in ('e', 'h', 'u') or source_id not in source_ids):
            raise ValueError(f'{grasp_path}: invalid grasp coordinates, owner, difficulty, or object class')
        points = list(zip(values[::2], values[1::2]))
        if any(math.dist(points[i], points[(i+1)%4]) <= 0 for i in range(4)):
            raise ValueError(f'{grasp_path}: degenerate grasp edge')
        if owner_classes is not None and owner_classes.get(owner) != source_id:
            raise ValueError(f'{grasp_path}: grasp owner/class disagrees with object rows')
        # Preserve corners, difficulty, and original instance IDs. Only the semantic ID changes.
        row[-1] = mapped
    stringify = lambda rows: ''.join(' '.join(row)+'\n' for row in rows)
    return stringify(labels), stringify(adjacency), stringify(grasps), count, len(grasps)


@dataclass
class Sample:
    source: str
    image: Path
    split: str
    labels: str
    adjacency: str
    grasps: str
    objects: int
    grasp_count: int
    digest: str
    metadata: dict


def audit(roots, overrides=None, aliases=None, grasp_policy='auto', duplicate_policy='error'):
    overrides = overrides or {}
    aliases = {name_key(k): ' '.join(v.split()) for k, v in (aliases or {}).items()}
    if any(not k or not v for k, v in aliases.items()):
        raise ValueError('Aliases must map nonempty source names to canonical names')
    # VMRD IDs are stable regardless of input dataset order.
    classes = list(CLASSES)
    class_ids = {name_key(name): i for i, name in enumerate(classes)}
    for name in CLASSES:
        if name_key(name) in aliases and name_key(aliases[name_key(name)]) != name_key(name):
            raise ValueError(f'Aliases must not rename VMRD class {name!r}')
    sources, samples, fingerprints = [], [], {}
    duplicates = []
    all_grasps = True
    used_names = set()
    for root in roots:
        names, registry_path = source_classes(root, overrides.get(root))
        source_name = root.name
        suffix = 2
        while source_name in used_names:
            source_name = f'{root.name}_{suffix}'
            suffix += 1
        used_names.add(source_name)
        mapping = {}
        for i, name in enumerate(names):
            target = aliases.get(name_key(name), name)
            key = name_key(target)
            if key not in class_ids:
                class_ids[key] = len(classes)
                classes.append(target)
            mapping[i] = class_ids[key]
        manifest_path = root / 'manifest.json'
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if not isinstance(manifest, dict) or not isinstance(manifest.get('samples', {}), dict):
            raise ValueError(f'{manifest_path}: expected a manifest with a samples object')
        tasks = manifest.get('tasks', {})
        if not isinstance(tasks, dict) or ('grasps' in tasks and type(tasks['grasps']) is not bool):
            raise ValueError(f'{manifest_path}: tasks.grasps must be true or false')
        if tasks.get('grasps') is False:
            all_grasps = False
            if grasp_policy == 'require':
                raise ValueError(f'{root}: explicitly exported without grasp supervision')
        source_counts = {split: {'images': 0, 'objects': 0, 'grasps': 0} for split in SPLITS}
        for split in SPLITS:
            image_folder = root / 'images' / split
            if not image_folder.is_dir() and split != 'test':
                raise ValueError(f'Missing image directory: {image_folder}')
            images = sorted((p for p in (image_folder.iterdir() if image_folder.is_dir() else []) if p.suffix.lower() in IMAGE_SUFFIXES),
                            key=lambda p: (p.stem, p.name))
            stems = set()
            for image in images:
                if image.stem in stems:
                    raise ValueError(f'{image_folder}: duplicate image stem {image.stem}')
                stems.add(image.stem)
                entry = manifest.get('samples', {}).get(image.stem, {})
                if not isinstance(entry, dict):
                    raise ValueError(f'{manifest_path}: sample metadata must be an object for {image.name}')
                if entry.get('split', split) != split:
                    raise ValueError(f'{manifest_path}: split mismatch for {image.name}')
                try:
                    label, adj, grasp, objects, grasp_count = annotations(root, split, image, mapping, entry)
                except (ValueError, OSError) as exc:
                    raise ValueError(f'{source_name}/{split}/{image.name}: {exc}') from exc
                if not grasp_count:
                    all_grasps = False
                    if grasp_policy == 'require':
                        raise ValueError(f'{image}: no grasp annotations')
                digest = hashlib.sha256()
                with image.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                digest = digest.hexdigest()
                identity = {'source': source_name, 'split': split, 'image': image.name}
                previous = fingerprints.setdefault(digest, [])
                for other in previous:
                    if other['split'] != split:
                        duplicates.append({'first': other, 'second': identity, 'sha256': digest})
                previous.append(identity)
                samples.append(Sample(source_name, image, split, label, adj, grasp, objects, grasp_count, digest, entry))
                source_counts[split]['images'] += 1
                source_counts[split]['objects'] += objects
                source_counts[split]['grasps'] += grasp_count
            # Reject orphan annotations rather than silently losing source samples.
            for folder in ('labels', 'adj', 'grasps'):
                orphans = {p.stem for p in (root / folder / split).glob('*.txt')} - stems
                if orphans:
                    raise ValueError(f'{root}/{folder}/{split}: orphan annotations {sorted(orphans)[:5]}')
        if not sum(info['images'] for info in source_counts.values()):
            raise ValueError(f'{root}: dataset contains no images')
        sources.append({'id': source_name, 'root': str(root), 'class_registry': registry_path,
                        'classes': names, 'class_id_map': mapping, 'counts': source_counts,
                        'declared_tasks': tasks})
    if duplicates and duplicate_policy == 'error':
        pair = duplicates[0]
        raise ValueError(f'Identical image occurs in different splits: {pair["first"]} and {pair["second"]}. '
                         'Use --cross-split-duplicates prefer-heldout to retain test/validation copies, or keep.')
    excluded = []
    if duplicate_policy in ('prefer-val', 'prefer-heldout'):
        priority = {'train': 0, 'val': 1, 'test': 2}
        best = {}
        for sample in samples:
            best[sample.digest] = max(best.get(sample.digest, -1), priority[sample.split])
        retained = []
        for sample in samples:
            if priority[sample.split] < best[sample.digest]:
                excluded.append({'source': sample.source, 'image': sample.image.name, 'split': sample.split,
                                 'reason': 'identical image retained in a higher-priority held-out split', 'sha256': sample.digest})
            else:
                retained.append(sample)
        samples = retained
    for source in sources:
        source['input_counts'] = source['counts']
        source['counts'] = {
            split: {'images': sum(s.source == source['id'] and s.split == split for s in samples),
                    'objects': sum(s.objects for s in samples if s.source == source['id'] and s.split == split),
                    'grasps': sum(s.grasp_count for s in samples if s.source == source['id'] and s.split == split)}
            for split in SPLITS}
    if not samples or any(not any(s.split == split for s in samples) for split in ('train', 'val')):
        raise ValueError('Combined dataset must have at least one image in both train and val')
    counts = {split: {'images': sum(s.split == split for s in samples),
                      'objects': sum(s.objects for s in samples if s.split == split),
                      'grasps': sum(s.grasp_count for s in samples if s.split == split)} for split in SPLITS}
    report = {'version': 1, 'operation': 'merge', 'classes': classes, 'sources': sources,
              'counts': counts, 'tasks': {'grasps': all_grasps and grasp_policy != 'disable'},
              'grasp_policy': grasp_policy, 'cross_split_duplicate_policy': duplicate_policy,
              'cross_split_duplicates': duplicates, 'excluded_samples': excluded,
              'same_split_duplicate_copies': sum(max(0, sum(i['split'] == split for i in entries)-1)
                                                 for entries in fingerprints.values() for split in SPLITS),
              'samples': {}}
    return samples, report


def write_dataset(output, samples, report):
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.merging-', dir=output.parent))
    try:
        for folder in FOLDERS:
            for split in SPLITS:
                (staging / folder / split).mkdir(parents=True)
        for index, sample in enumerate(samples, 1):
            stem = f'{index:06d}'
            suffix = sample.image.suffix.lower()
            if suffix not in ('.png', '.jpg', '.jpeg'):
                suffix = '.png'
                with Image.open(sample.image) as image:
                    image.convert('RGB').save(staging / 'images' / sample.split / (stem+suffix))
            else:
                shutil.copy2(sample.image, staging / 'images' / sample.split / (stem+suffix))
            for folder, text in (('labels', sample.labels), ('adj', sample.adjacency), ('grasps', sample.grasps)):
                (staging / folder / sample.split / (stem+'.txt')).write_text(text)
            report['samples'][stem] = {'source': sample.source, 'image': sample.image.name,
                                       'split': sample.split, 'source_sha256': sample.digest,
                                       'original_annotation': sample.metadata}
        (staging / 'classes.json').write_text(json.dumps(report['classes'], indent=2)+'\n')
        (staging / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        if output.exists() or output.is_symlink():
            raise ValueError(f'Output appeared during merge: {output}')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True, help='Exported dataset directories')
    parser.add_argument('--output', type=Path, required=True, help='New combined dataset directory')
    parser.add_argument('--class-map', action='append', default=[], metavar='DATASET=JSON',
                        help='Override a source registry; JSON is a class-name list or name-to-ID dictionary')
    parser.add_argument('--aliases', type=Path, help='Optional JSON source-name to canonical-name aliases')
    parser.add_argument('--grasp-policy', choices=('auto', 'disable', 'require'), default='auto',
                        help='auto disables grasp training if any input lacks supervision; all grasp files are retained')
    parser.add_argument('--cross-split-duplicates', choices=('error', 'prefer-heldout', 'prefer-val', 'keep'), default='error',
                        help='prefer-heldout retains test over val over train; prefer-val is a compatibility alias')
    parser.add_argument('--dry-run', action='store_true', help='Audit without writing a dataset')
    args = parser.parse_args(argv)
    roots = [p.expanduser().resolve() for p in args.inputs]
    output = args.output.expanduser().resolve()
    if len(set(roots)) != len(roots):
        raise ValueError('The same input dataset was supplied more than once')
    if any(output == root or root in output.parents or output in root.parents for root in roots):
        raise ValueError('Output and input directories must not contain one another')
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    overrides = {}
    for specification in args.class_map:
        source, separator, filename = specification.partition('=')
        if not separator:
            raise ValueError('--class-map requires DATASET=JSON')
        root = Path(source).expanduser().resolve()
        if root not in roots or root in overrides:
            raise ValueError(f'Class-map source is not a unique input: {source}')
        overrides[root] = Path(filename).expanduser().resolve()
    aliases = read_json(args.aliases) if args.aliases else {}
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in aliases.items()):
        raise ValueError('--aliases must be a JSON object mapping class names to class names')
    samples, report = audit(roots, overrides, aliases, args.grasp_policy, args.cross_split_duplicates)
    print(f'{len(samples)} images, {len(report["classes"])} classes')
    for source in report['sources']:
        print(f'  {source["id"]}: '+', '.join(f'{split}={source["counts"][split]["images"]}' for split in SPLITS))
    print('Grasp training: '+('enabled' if report['tasks']['grasps'] else 'disabled (existing grasps retained)'))
    if report['excluded_samples']:
        print(f'Excluded {len(report["excluded_samples"])} lower-priority duplicate copies (recorded in manifest).')
    elif report['cross_split_duplicates']:
        print(f'WARNING: {len(report["cross_split_duplicates"])} identical-image pairs span splits')
    if args.dry_run:
        print('Audit passed; no dataset written.')
    else:
        write_dataset(output, samples, report)
        print(f'Combined dataset: {output}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
