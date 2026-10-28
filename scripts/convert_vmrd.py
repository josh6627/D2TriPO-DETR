#!/usr/bin/env python3
"""Convert original VMRD V2 annotations without repairing ambiguous source data.

Only Pillow and the Python standard library are needed. Exit 2 means some source
samples were withheld; the report records their IDs and reasons. No model imports.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from PIL import Image


# Zero-based order used by robot/inference.py:draw_rectangle (not alphabetical).
CLASSES = (
    "banana", "watch", "pliers", "toothbrush", "wrench", "towel", "knife",
    "badminton", "tape", "charger", "pen", "box", "umbrella", "apple",
    "wallet", "shaver", "paper", "cup", "mobile phone", "wrist developer",
    "bottle", "headset", "toothpaste", "glasses", "remote controller", "card",
    "mouse", "cans", "screwdriver", "notebook", "stapler",
)
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
OUTPUTS = ("images", "labels", "adj", "grasps")


def read_split(path):
    ids = path.read_text().splitlines()
    if not ids or any(not re.fullmatch(r"[0-9]+", i) for i in ids):
        raise ValueError(f"Empty split or invalid image ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def parse_sample(root, image_id):
    errors, warnings = [], []
    paths = [root / directory / f"{image_id}.{ext}" for directory, ext in
             (("JPEGImages", "jpg"), ("Annotations", "xml"), ("Grasps", "txt"))]
    for path in paths:
        if not path.is_file():
            errors.append(f"Missing source file: {path.relative_to(root)}")
    if errors:
        return None, errors, warnings
    image_path, xml_path, grasp_path = paths
    annotation = ET.parse(xml_path).getroot()
    width = int(annotation.findtext("size/width"))
    height = int(annotation.findtext("size/height"))
    with Image.open(image_path) as image:
        if image.size != (width, height) or min(width, height) <= 0:
            errors.append(f"Image dimensions {image.size} disagree with XML {(width, height)}")
    objects = annotation.findall("object")
    if not objects:
        errors.append("No objects")
    indices = [int(obj.findtext("index")) for obj in objects]
    duplicates = sorted(i for i, count in Counter(indices).items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate object indices: {duplicates}; associations are ambiguous")
    if any(i <= 0 for i in indices):
        errors.append(f"Nonpositive object indices: {indices}")
    rows = {index: row for row, index in enumerate(indices)}
    labels, class_by_index = [], {}
    father_edges, child_edges = set(), set()
    for row, obj in enumerate(objects):
        index = indices[row]
        name = obj.findtext("name")
        class_id = CLASS_IDS.get(name)
        if class_id is None:
            errors.append(f"Unknown object class {name!r} at XML row {row + 1}")
        class_by_index[index] = class_id
        coords = [obj.findtext(f"bndbox/{key}") for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = map(float, coords)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"Invalid box for object {index}: {coords}")
        # Exactly seven fields: class xmin ymin xmax ymax width height.
        # Preserve the XML numbers verbatim: no -1, +1, normalization or clipping.
        labels.append(" ".join([str(class_id), *coords, str(width), str(height)]))
        for relation, edges in (("father", father_edges), ("children", child_edges)):
            refs = [int(n.text) for n in obj.findall(f"{relation}/num")]
            if len(refs) != len(set(refs)):
                warnings.append(f"Repeated {relation} references on object {index}: {refs}; "
                                "identical edges collapse in a binary adjacency matrix")
            for ref in refs:
                if ref not in rows or ref == index:
                    errors.append(f"Invalid {relation} reference {index} -> {ref}")
                # Above -> below: a child is grasped before its father.
                edges.add((index, ref) if relation == "father" else (ref, index))
    if father_edges != child_edges:
        errors.append(f"Nonreciprocal relationships: father-only {sorted(father_edges - child_edges)}, "
                      f"children-only {sorted(child_edges - father_edges)}")
    adj = [[0] * len(objects) for _ in objects]
    if not duplicates and all(a in rows and b in rows for a, b in father_edges):
        for a, b in father_edges:
            adj[rows[a]][rows[b]] = 1
        # A manipulation order must be acyclic. Do not add transitive edges.
        remaining = set(range(len(objects)))
        while remaining:
            leaves = {i for i in remaining if not any(adj[i][j] for j in remaining)}
            if not leaves:
                errors.append("Cyclic manipulation relationships")
                break
            remaining -= leaves
    grasps = []
    for line_number, line in enumerate(grasp_path.read_text().splitlines(), 1):
        parts = line.split()
        prefix = f"Grasp line {line_number}"
        if len(parts) != 10:
            errors.append(f"{prefix}: expected 10 source fields, found {len(parts)}")
            continue
        try:
            coords = list(map(float, parts[:8]))
            index = int(parts[8])
        except ValueError:
            errors.append(f"{prefix}: invalid numeric fields")
            continue
        if not all(math.isfinite(v) for v in coords):
            errors.append(f"{prefix}: nonfinite coordinates")
        if index not in rows:
            errors.append(f"{prefix}: unknown object index {index}; XML indices {indices}")
        if parts[9] not in ("e", "h"):
            errors.append(f"{prefix}: unknown difficulty {parts[9]!r}; no e/h mapping established")
        if any(not 0 <= v <= (width if k % 2 == 0 else height) for k, v in enumerate(coords)):
            warnings.append(f"{prefix}: corner outside image; preserved without clipping")
        if any(math.hypot(coords[2*k] - coords[2*((k+1)%4)],
                          coords[2*k+1] - coords[2*((k+1)%4)+1]) == 0 for k in range(4)):
            errors.append(f"{prefix}: zero-length rectangle edge")
        # First 8 + difficulty (-3) + original instance ID (-2) + class (-1).
        grasps.append(" ".join([*parts[:8], parts[9], parts[8], str(class_by_index.get(index))]))
    if not grasps:
        errors.append("No grasps")
    if errors:
        return None, errors, warnings
    return {
        "labels": "\n".join(labels) + "\n",
        "adj": "\n".join(" ".join(map(str, row)) for row in adj) + "\n",
        "grasps": "\n".join(grasps) + "\n",
        "object_indices": indices,
        "object_count": len(objects), "grasp_count": len(grasps),
    }, errors, warnings


def check_existing(path, expected):
    if path.is_symlink() or not path.is_file() or path.read_text() != expected:
        raise ValueError(f"Missing or differing converted annotation: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/vmrd"))
    parser.add_argument("--check", action="store_true", help="Verify existing outputs without writing")
    args = parser.parse_args()
    root = args.root.resolve()
    splits = {split: read_split(root / "ImageSets/Main" / filename) for split, filename in
              (("train", "trainval.txt"), ("val", "test.txt"))}
    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise ValueError(f"Train/val overlap: {sorted(overlap)}")
    report = {"format_version": 1, "class_ids": CLASS_IDS, "splits": {}, "blocked": {}, "warnings": {}}
    samples = {}
    for split, ids in splits.items():
        converted = []
        counts = Counter()
        for image_id in ids:
            try:
                sample, errors, warnings = parse_sample(root, image_id)
            except (ValueError, TypeError, OSError, ET.ParseError) as exc:
                sample, errors, warnings = None, [str(exc)], []
            key = f"{split}/{image_id}"
            if warnings:
                report["warnings"][key] = warnings
            if errors:
                report["blocked"][key] = errors
                continue
            samples[(split, image_id)] = sample
            converted.append(image_id)
            counts["objects"] += sample["object_count"]
            counts["grasps"] += sample["grasp_count"]
        report["splits"][split] = {"expected": len(ids), "converted": len(converted),
                                    "withheld": len(ids) - len(converted), "ids": converted, **counts}

    # Preflight every destination before writing. Existing files must match;
    # original files and unexpected destination content are never overwritten.
    for split in splits:
        valid_ids = set(report["splits"][split]["ids"])
        for directory in OUTPUTS:
            folder = root / directory / split
            if (root / directory).is_symlink() or folder.is_symlink():
                raise ValueError(f"Refusing output-directory symlink: {folder}")
            if folder.exists():
                expected = {i + (".jpg" if directory == "images" else ".txt") for i in valid_ids}
                extras = {p.name for p in folder.iterdir()} - expected
                if extras:
                    raise ValueError(f"Unexpected output files in {folder}: {sorted(extras)}")
    for (split, image_id), sample in samples.items():
        for directory in OUTPUTS:
            path = root / directory / split / (image_id + (".jpg" if directory == "images" else ".txt"))
            if directory == "images":
                source = root / "JPEGImages" / f"{image_id}.jpg"
                if os.path.lexists(path):
                    if not (path.is_symlink() and path.resolve() == source):
                        raise ValueError(f"Expected image symlink to {source}: {path}")
                elif args.check:
                    raise ValueError(f"Missing image link: {path}")
            elif os.path.lexists(path) or args.check:
                check_existing(path, sample[directory])
    if not args.check:
        for split in splits:
            for directory in OUTPUTS:
                (root / directory / split).mkdir(parents=True, exist_ok=True)
        for (split, image_id), sample in samples.items():
            for directory in OUTPUTS:
                path = root / directory / split / (image_id + (".jpg" if directory == "images" else ".txt"))
                if os.path.lexists(path):
                    continue
                if directory == "images":
                    source = root / "JPEGImages" / f"{image_id}.jpg"
                    path.symlink_to(os.path.relpath(source, path.parent))
                else:
                    with path.open("x") as handle:
                        handle.write(sample[directory])
        report_path = root / "conversion_report.json"
        if report_path.is_symlink():
            raise ValueError(f"Refusing report symlink: {report_path}")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    # Re-read every emitted annotation and check every image link.
    for (split, image_id), sample in samples.items():
        for directory in ("labels", "adj", "grasps"):
            check_existing(root / directory / split / f"{image_id}.txt", sample[directory])
        link = root / "images" / split / f"{image_id}.jpg"
        if not link.is_symlink() or link.resolve() != root / "JPEGImages" / f"{image_id}.jpg":
            raise ValueError(f"Invalid image link: {link}")
    if args.check:
        stored = json.loads((root / "conversion_report.json").read_text())
        if stored != report:
            raise ValueError("Conversion report differs from current source audit")
    for split, result in report["splits"].items():
        print(f"{split}: {result['converted']}/{result['expected']} converted, "
              f"{result['withheld']} withheld, {result['objects']} objects, {result['grasps']} grasps")
    print(f"Integrity checks passed for all converted samples. Warning samples: {len(report['warnings'])}.")
    if report["blocked"]:
        print(f"INCOMPLETE: {len(report['blocked'])} ambiguous samples withheld; see {root / 'conversion_report.json'}")
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        sys.exit(f"Conversion stopped: {exc}")
