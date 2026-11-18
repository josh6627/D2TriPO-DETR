"""Persistent manual annotations and strict D2TriPO-DETR export (no model imports)."""
import json
import math
from pathlib import Path
import random
import shutil

try:
    from .convert_vmrd import CLASSES
except ImportError:
    from convert_vmrd import CLASSES

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = ('train', 'val', 'test')


def split_fractions(val_fraction=.1, test_fraction=.2):
    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (val_fraction, test_fraction))
            or val_fraction+test_fraction > 1):
        raise ValueError('Train, validation, and test fractions must be nonnegative and sum to one.')
    return {'train': 1-(val_fraction+test_fraction), 'val': val_fraction, 'test': test_fraction}


def test_target(total, fraction):
    return max(1, math.floor(total*fraction+.5)) if total and fraction > 0 else 0


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def grasp_rectangle(a, b, half_height):
    """a/b are opposing jaw midpoints; corners 2/3 face b, as in the loader."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1 or half_height < 0.5:
        raise ValueError('Grasp opening and height must be at least one pixel.')
    nx, ny = -dy / length * half_height, dx / length * half_height
    return [[a[0]-nx, a[1]-ny], [b[0]-nx, b[1]-ny],
            [b[0]+nx, b[1]+ny], [a[0]+nx, a[1]+ny]]


def validate(sample, classes, size, complete=False, require_grasps=True):
    width, height = size
    objects, grasps, edges = sample['objects'], sample['grasps'], sample['edges']
    ids = [o['id'] for o in objects]
    if len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError('Object instance IDs must be unique positive integers.')
    for obj in objects:
        if type(obj['class_id']) is not int or not 0 <= obj['class_id'] < len(classes):
            raise ValueError('Invalid object class.')
        x1, y1, x2, y2 = obj['box']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Boxes must have positive area and stay inside the image.')
    for grasp in grasps:
        if grasp['object_id'] not in ids or grasp['difficulty'] not in ('e', 'h'):
            raise ValueError('Each grasp needs an existing object and e/h difficulty.')
        points = grasp['points']
        if len(points) != 4 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError('Grasps need four finite 2D corners.')
        vectors = [[points[(i+1)%4][j]-points[i][j] for j in range(2)] for i in range(4)]
        lengths = [math.hypot(*v) for v in vectors]
        if min(lengths) < 1:
            raise ValueError('Grasp edges must be at least one pixel.')
        if (abs(sum(vectors[0][j]*vectors[1][j] for j in range(2))) > 1e-5*lengths[0]*lengths[1]
                or any(abs(vectors[0][j]+vectors[2][j]) > 1e-5 for j in range(2))
                or any(abs(vectors[1][j]+vectors[3][j]) > 1e-5 for j in range(2))):
            raise ValueError('Grasp corners must form an ordered rectangle.')
    graph = {i: set() for i in ids}
    for above, below in edges:
        if above not in graph or below not in graph or above == below:
            raise ValueError('Relationships need two different existing objects.')
        graph[above].add(below)
    remaining = set(ids)
    while remaining:
        leaves = {i for i in remaining if not (graph[i] & remaining)}
        if not leaves:
            raise ValueError('Relationships form a cycle. Use above → below only.')
        remaining -= leaves
    if complete:
        if not objects:
            raise ValueError('Training requires at least one object.')
        missing = set(ids) - {g['object_id'] for g in grasps}
        if require_grasps and missing:
            raise ValueError(f'Add at least one grasp for each object; missing: {sorted(missing)}')
        if not sample.get('relations_reviewed'):
            raise ValueError('Confirm that relationships have been reviewed (including no overlaps).')


class Project:
    def __init__(self, images, path, exclude=('density_map.png',), require_grasps=None):
        self.images = Path(images).resolve()
        self.path = Path(path).resolve()
        self.files = sorted((p.name for p in self.images.iterdir()
                             if p.is_file() and p.suffix.lower() in EXTENSIONS and p.name not in exclude),
                            key=lambda n: (0, int(Path(n).stem), n) if Path(n).stem.isdigit() else (1, 0, n))
        if not self.files:
            raise ValueError(f'No images in {self.images}')
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get('version') != 1 or self.data['image_root'] != str(self.images):
                raise ValueError('Project version or image folder does not match.')
            classes = self.data['classes']
            if (classes[:len(CLASSES)] != list(CLASSES) or len(set(classes)) != len(classes)
                    or any(not isinstance(c, str) or not c.strip() for c in classes)):
                raise ValueError('Class registry must preserve VMRD IDs 0–30 and unique names.')
        else:
            self.data = {'version': 1, 'image_root': str(self.images),
                         'classes': list(CLASSES), 'samples': {}}
        self.data.setdefault('require_grasps', True)
        self.data.setdefault('split_fractions', split_fractions())
        self.data.setdefault('test_images', [])
        if require_grasps is not None:
            self.set_require_grasps(require_grasps)

    @property
    def require_grasps(self):
        return self.data['require_grasps']

    def set_require_grasps(self, required):
        self.data['require_grasps'] = bool(required)
        if required:
            for sample in self.data['samples'].values():
                if ({o['id'] for o in sample['objects']}
                        - {g['object_id'] for g in sample['grasps']}):
                    sample['complete'] = False
        self.save()

    @property
    def classes(self):
        return self.data['classes']

    def sample(self, name):
        return self.data['samples'].setdefault(name, {
            'objects': [], 'grasps': [], 'edges': [], 'relations_reviewed': False, 'complete': False})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.data)

    def set_split_fractions(self, val_fraction, test_fraction):
        self.data['split_fractions'] = split_fractions(val_fraction, test_fraction)
        self.save()

    def set_test_image(self, name, selected):
        if name not in self.files:
            raise ValueError(f'Unknown image: {name}')
        names = set(self.data['test_images'])
        names.add(name) if selected else names.discard(name)
        self.data['test_images'] = sorted(names)
        self.save()

    def test_progress(self, fraction=None):
        complete = [n for n in self.files if self.sample(n).get('complete')]
        chosen = set(self.data['test_images'])
        fraction = self.data['split_fractions']['test'] if fraction is None else fraction
        selected = sum(n in chosen for n in complete)
        return {'completed': len(complete), 'selected': selected,
                'target': test_target(len(complete), fraction), 'fraction': fraction,
                'percentage': 100*selected/len(complete) if complete else 0,
                'planned': sum(n in chosen for n in self.files)}

    def add_class(self, name):
        name = name.strip()
        if not name or name.casefold() in {c.casefold() for c in self.classes}:
            raise ValueError('Enter a new, nonempty class name.')
        self.classes.append(name)
        self.save()
        return len(self.classes)-1

    def export(self, output, val_fraction=None, seed=42, test_fraction=None):
        from PIL import Image
        output = Path(output).resolve()
        if output.exists():
            raise ValueError('Choose a new export directory; existing exports are never overwritten.')
        settings = self.data['split_fractions']
        fractions = split_fractions(settings['val'] if val_fraction is None else val_fraction,
                                    settings['test'] if test_fraction is None else test_fraction)
        names = [n for n in self.files if self.sample(n).get('complete')]
        required = sum(f > 0 for f in fractions.values())
        if len(names) < required:
            minimum = 'three' if required == 3 else str(required)
            raise ValueError(f'Mark at least {minimum} images complete for the nonzero splits.')
        test_names = set(self.data['test_images']) & set(names)
        target = test_target(len(names), fractions['test'])
        if len(test_names) != target:
            raise ValueError(f'Choose exactly {target} completed test images for the '
                             f'{fractions["test"]:.0%} target ({len(names)} completed images); '
                             f'currently selected {len(test_names)}. Test images are never chosen automatically.')
        non_test_splits = sum(fractions[s] > 0 for s in ('train', 'val'))
        if len(names)-len(test_names) < non_test_splits:
            raise ValueError(f'Leave at least {non_test_splits} completed images outside test '
                             'for the nonzero training/validation splits.')
        sizes = {}
        for name in names:
            with Image.open(self.images / name) as im:
                sizes[name] = im.size
            try:
                validate(self.sample(name), self.classes, sizes[name], complete=True,
                         require_grasps=self.require_grasps)
            except ValueError as exc:
                raise ValueError(f'{name}: {exc}') from exc
        shuffled = [n for n in names if n not in test_names]
        random.Random(seed).shuffle(shuffled)
        val_count = (0 if fractions['val'] == 0 else len(shuffled) if fractions['train'] == 0
                     else max(1, min(len(shuffled)-1, math.floor(len(names)*fractions['val']+.5))))
        val_names = set(shuffled[:val_count])
        counts = {'train': len(shuffled)-val_count, 'val': val_count, 'test': len(test_names)}
        manifest = {'version': 1, 'source': str(self.path), 'seed': seed,
                    'val_fraction': fractions['val'], 'test_fraction': fractions['test'],
                    'split_fractions': fractions, 'split_counts': counts,
                    'tasks': {'grasps': self.require_grasps}, 'samples': {}}
        # Build beside the destination, then rename so failures never leave a usable partial export.
        staging = output.with_name(output.name + '.exporting')
        if staging.exists():
            raise ValueError(f'Staging folder already exists: {staging}')
        try:
            for directory in ('images', 'labels', 'adj', 'grasps'):
                for split in SPLITS:
                    (staging / directory / split).mkdir(parents=True, exist_ok=True)
            atomic_json(staging / 'classes.json', self.classes)
            for index, name in enumerate(names, 1):
                sample = self.sample(name)
                split = 'test' if name in test_names else 'val' if name in val_names else 'train'
                stem = f'{index:06d}'
                width, height = sizes[name]
                # Copy pixels for a portable export, including non-PNG source formats.
                with Image.open(self.images / name) as im:
                    im.convert('RGB').save(staging / 'images' / split / f'{stem}.png')
                objects = sample['objects']
                rows = {o['id']: i for i, o in enumerate(objects)}
                class_ids = {o['id']: o['class_id'] for o in objects}
                labels = [' '.join(map(str, [o['class_id'], *o['box'], width, height])) for o in objects]
                adj = [[0]*len(objects) for _ in objects]
                for a, b in sample['edges']:
                    adj[rows[a]][rows[b]] = 1
                grasps = [' '.join(map(str, [*(v for p in g['points'] for v in p),
                                             g['difficulty'], g['object_id'], class_ids[g['object_id']]]))
                          for g in sample['grasps']]
                for folder, lines in (('labels', labels), ('adj', [' '.join(map(str, r)) for r in adj]),
                                      ('grasps', grasps)):
                    (staging / folder / split / f'{stem}.txt').write_text('\n'.join(lines)+('\n' if lines else ''))
                manifest['samples'][stem] = {'image': name, 'split': split,
                                            'object_ids_in_row_order': list(rows)}
            atomic_json(staging / 'manifest.json', manifest)
            staging.rename(output)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return output
