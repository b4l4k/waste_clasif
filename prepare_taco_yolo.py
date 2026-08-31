import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_class_map(path, categories):
    """Map TACO's original category names to merged target names."""
    if path is None:
        return {category["name"]: category["name"] for category in categories}

    with path.open(newline="", encoding="utf-8") as file:
        return {source: target for source, target in csv.reader(file)}


def link_or_copy(source, destination):
    """Avoid duplicating images when hard links are supported."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations.json"),
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional TACO CSV class map, such as map_10.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("taco_yolo"),
    )
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"{args.output} is not empty. Choose a new output directory."
        )

    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = sorted(dataset["categories"], key=lambda item: item["id"])

    category_name_by_id = {
        category["id"]: category["name"]
        for category in categories
    }

    source_to_target = read_class_map(args.class_map, categories)

    # Preserve a stable target-class order.
    target_names = []
    for category in categories:
        source_name = category["name"]
        target_name = source_to_target.get(source_name)

        if target_name is None or target_name == "Background":
            continue

        if target_name not in target_names:
            target_names.append(target_name)

    target_id_by_name = {
        name: class_id
        for class_id, name in enumerate(target_names)
    }

    annotations_by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    # Deterministic 80/10/10 split by default.
    image_ids = [image["id"] for image in dataset["images"]]
    random.Random(args.seed).shuffle(image_ids)

    test_count = round(len(image_ids) * args.test)
    validation_count = round(len(image_ids) * args.validation)

    test_ids = set(image_ids[:test_count])
    validation_ids = set(
        image_ids[test_count:test_count + validation_count]
    )

    def get_split(image_id):
        if image_id in test_ids:
            return "test"
        if image_id in validation_ids:
            return "val"
        return "train"

    counts = defaultdict(lambda: defaultdict(int))
    dataset_root = args.annotations.parent

    for image in dataset["images"]:
        split = get_split(image["id"])

        source_image = dataset_root / image["file_name"]
        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing {source_image}. Run: python download.py"
            )

        relative_path = Path(image["file_name"])

        destination_image = (
            args.output / "images" / split / relative_path
        )
        destination_label = (
            args.output
            / "labels"
            / split
            / relative_path.with_suffix(".txt")
        )

        link_or_copy(source_image, destination_image)
        destination_label.parent.mkdir(parents=True, exist_ok=True)

        image_width = float(image["width"])
        image_height = float(image["height"])
        label_lines = []

        for annotation in annotations_by_image.get(image["id"], []):
            if annotation.get("iscrowd", 0):
                continue

            source_name = category_name_by_id[annotation["category_id"]]
            target_name = source_to_target.get(source_name)

            if target_name is None or target_name == "Background":
                continue

            # COCO bbox: x_min, y_min, width, height in pixels.
            x, y, width, height = map(float, annotation["bbox"])

            # Keep the box inside the image.
            x1 = max(0.0, min(image_width, x))
            y1 = max(0.0, min(image_height, y))
            x2 = max(0.0, min(image_width, x + width))
            y2 = max(0.0, min(image_height, y + height))

            if x2 <= x1 or y2 <= y1:
                continue

            # Convert to normalized YOLO coordinates.
            center_x = ((x1 + x2) / 2.0) / image_width
            center_y = ((y1 + y2) / 2.0) / image_height
            normalized_width = (x2 - x1) / image_width
            normalized_height = (y2 - y1) / image_height

            target_id = target_id_by_name[target_name]

            label_lines.append(
                f"{target_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

            counts[split][target_name] += 1

        destination_label.write_text(
            "\n".join(label_lines),
            encoding="utf-8",
        )

    # Generate the Ultralytics dataset configuration.
    yaml_lines = [
        f"path: {json.dumps(args.output.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]

    for class_id, name in enumerate(target_names):
        yaml_lines.append(f"  {class_id}: {json.dumps(name)}")

    (args.output / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Created dataset: {args.output.resolve()}")
    print(f"Classes ({len(target_names)}):")

    for class_id, name in enumerate(target_names):
        print(f"  {class_id}: {name}")

    for split in ("train", "val", "test"):
        total = sum(counts[split].values())
        print(f"{split}: {total} labeled objects")


if __name__ == "__main__":
    main()