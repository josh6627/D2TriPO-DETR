"""Class configuration shared by training and model builders."""
import json
from pathlib import Path


def class_id_mapping(dataset_root, target_names):
    """Map a labeled dataset's class IDs into the checkpoint's class order."""
    root = Path(dataset_root)
    path = root / 'classes.json'
    if path.is_file():
        source_names = json.loads(path.read_text())
    else:
        path = root / 'conversion_report.json'
        if not path.is_file():
            return None  # Legacy datasets without a registry use model IDs directly.
        source_names = json.loads(path.read_text()).get('class_ids')
    if isinstance(source_names, dict):
        if (any(type(i) is not int for i in source_names.values())
                or set(source_names.values()) != set(range(len(source_names)))):
            raise ValueError(f'{path}: expected contiguous class IDs starting at zero')
        source_names = [name for name, _ in sorted(source_names.items(), key=lambda item: item[1])]
    key = lambda name: ' '.join(name.split()).casefold()
    for names, description in ((source_names, str(path)), (target_names, 'checkpoint class names')):
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(n, str) or not n.strip() for n in names)
                or len({key(n) for n in names}) != len(names)):
            raise ValueError(f'{description}: expected unique, nonempty class names')
    targets = {key(name): i for i, name in enumerate(target_names)}
    # Unrepresented source classes remain unmapped; report an error only if an
    # evaluated image uses one. Registries can include unused classes.
    return {i: targets[key(name)] for i, name in enumerate(source_names) if key(name) in targets}


def configure_classes(args):
    root = getattr(args, 'dataset_root', None)
    path = Path(root) / 'classes.json' if root else None
    names = None
    if path and path.is_file():
        names = json.loads(path.read_text())
        if (not isinstance(names, list) or not names
                or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != len(names)):
            raise ValueError(f'{path} must be a JSON list of unique, nonempty class names.')
    count = getattr(args, 'num_classes', None)
    if count is None:
        count = max(32, len(names)) if names else 32
    if type(count) is not int or count < 1 or (names and count < len(names)):
        raise ValueError('--num_classes must cover every ID in classes.json and be positive.')
    args.num_classes = count
    if names is not None:
        args.class_names = names
    return count


def get_num_classes(args):
    # Old checkpoints have no num_classes setting; their heads have 32 outputs.
    count = getattr(args, 'num_classes', None)
    return 32 if count is None else count


def configure_tasks(args):
    """Infer disabled grasp supervision from labeler exports unless explicitly set."""
    if getattr(args, 'no_grasps', None) is None:
        root = getattr(args, 'dataset_root', None)
        manifest = Path(root) / 'manifest.json' if root else None
        data = json.loads(manifest.read_text()) if manifest and manifest.is_file() else {}
        args.no_grasps = data.get('tasks', {}).get('grasps', True) is False
    if args.no_grasps and getattr(args, 'modelname', 'dab_deformable_detr') != 'dab_deformable_detr':
        raise ValueError('Use --modelname dab_deformable_detr for boxes/relationships training; dab_detr is grasp-only.')
    return args.no_grasps
