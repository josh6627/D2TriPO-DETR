#!/usr/bin/env python3
"""Create a NEW labeler project from model predictions, preserving class IDs."""
import argparse
import json
from pathlib import Path

from scripts.label_project import Project, validate
from scripts.convert_vmrd import CLASSES


def read_classes(path):
    data = json.loads(Path(path).read_text())
    names = data['classes'] if isinstance(data, dict) else data
    if (not isinstance(names, list) or names[:len(CLASSES)] != list(CLASSES)
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(n.casefold() for n in names)) != len(names)):
        raise ValueError('Class registry must preserve VMRD IDs and contain unique names.')
    return names


def check_classes(expected, actual):
    if actual is None or list(actual) != expected:
        raise ValueError('Checkpoint class_names must exactly match --classes-from, including '
                         'order. Refusing to guess, reorder, or add class IDs.')


def make_sample(prediction, classes, size):
    """Convert zero-based prediction IDs; keep strongest acyclic relationships."""
    sample = dict(objects=[], grasps=[], edges=[], relations_reviewed=False, complete=False)
    ids = {}
    skipped = 0
    for det in prediction['detections']:
        class_id = det['class_id']
        box = det['bbox_xyxy']
        if not 0 <= class_id < len(classes):
            skipped += 1  # Extra classifier channels have no semantic registry entry.
            continue
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            skipped += 1
            continue
        object_id = len(sample['objects']) + 1
        ids[det['id']] = object_id
        sample['objects'].append(dict(id=object_id, class_id=class_id, box=box))
    rejected_edges = 0
    for edge in sorted(prediction['relationships'], key=lambda e: e['confidence'], reverse=True):
        above, below = ids.get(edge['above_id']), ids.get(edge['below_id'])
        if above is None or below is None:
            rejected_edges += 1
            continue
        pair = [above, below]
        if pair in sample['edges']:
            continue
        sample['edges'].append(pair)
        try:
            validate(sample, classes, size, require_grasps=False)
        except ValueError:
            sample['edges'].pop()
            rejected_edges += 1
    validate(sample, classes, size, require_grasps=False)
    return sample, skipped, rejected_edges


def write_project(path, data):
    """Exclusive creation prevents replacing any existing manual annotations."""
    contents = json.dumps(data, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True,
                        help='Flat image folder, as required by the existing labeler')
    parser.add_argument('--project', type=Path, required=True, help='NEW output project.json')
    parser.add_argument('--classes-from', type=Path, default=Path('data/real_cluttered_lab/classes.json'),
                        help='Existing training classes.json or labeling project.json')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('checkpoints/real-cluttered-lab_finetune/checkpoint.pth'))
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--relation-threshold', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--limit', type=int, help='Trial run; remaining images stay empty/unreviewed')
    parser.add_argument('--device', default='cuda', help='CUDA device, e.g. cuda or cuda:0')
    parser.add_argument('--require-grasps', action='store_true',
                        help='Require manually added grasps for completion (none are predicted)')
    args = parser.parse_args()
    if not (0 <= args.score_threshold <= 1 and 0 <= args.relation_threshold <= 1):
        parser.error('Thresholds must be between 0 and 1')
    if args.topk < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--topk and --limit must be positive')
    if args.project.exists():
        parser.error('Output project already exists; choose a new path. Nothing was changed.')
    classes = read_classes(args.classes_from)
    project = Project(args.images, args.project)
    project.data['classes'] = classes.copy()
    project.data['require_grasps'] = args.require_grasps

    # Keep model dependencies out of project conversion and --help.
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('This model requires CUDA for deformable attention. Run in your '
                     'GPU environment; CPU inference is not supported by its compiled operator.')
    from PIL import Image
    from infer_unlabeled_dataset import decode_predictions, make_transform
    from models import build_dab_deformable_detr

    # Trusted local training checkpoint includes argparse settings.
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    check_classes(classes, getattr(checkpoint['args'], 'class_names', None))
    print(f'Verified all {len(classes)} class IDs against checkpoint.', flush=True)
    model, _, _ = build_dab_deformable_detr(checkpoint['args'])
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model = model.to(device).eval()
    transform = make_transform()
    files = project.files[:args.limit] if args.limit else project.files
    for index, name in enumerate(files, 1):
        with Image.open(project.images / name) as source:
            image = source.convert('RGB')
        with torch.inference_mode():
            outputs = model(transform(image).unsqueeze(0).to(device))
        prediction = decode_predictions(outputs, *image.size, args.score_threshold,
                                        args.relation_threshold, args.topk, class_names=classes)
        sample, skipped, rejected = make_sample(prediction, classes, image.size)
        project.data['samples'][name] = sample
        print(f'[{index}/{len(files)}] {name}: {len(sample["objects"])} objects, '
              f'{len(sample["edges"])} relationships; skipped {skipped} boxes and '
              f'{rejected} invalid/cyclic relationships', flush=True)
    project.data['prelabel_source'] = {
        'checkpoint': str(args.checkpoint.resolve()),
        'classes_from': str(args.classes_from.resolve()),
        'score_threshold': args.score_threshold, 'relation_threshold': args.relation_threshold,
    }
    write_project(args.project, project.data)
    print(f'Saved {args.project}. All predictions require review; no grasps were generated.')


if __name__ == '__main__':
    main()
