"""Class configuration shared by training and model builders."""
import json
from pathlib import Path


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
