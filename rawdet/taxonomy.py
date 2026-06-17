"""
Unified taxonomy for cross-sensor RAW detection.

Every dataset annotates a different class set. To compare methods fairly on a
single FROZEN COCO-pretrained detector, we (1) map each dataset's labels into
the COCO contiguous label space (the 80-class order the detector outputs) and
(2) for any leave-one-sensor-out fold, evaluate only on the classes the
participating datasets SHARE. Without this, a cross-dataset number is comparing
different label spaces and means nothing.

PASCALRAW and NOD class strings are concrete. LOD / ROD / AODRAW are filled with
plausible placeholders -- replace them with the exact strings from each
dataset's own class file. The evaluable set is then computed automatically.
"""

# COCO 80-class contiguous labels (the order mmdet detectors output).
# Only classes relevant to our datasets are listed; the int is the detector's label.
COCO_LABEL = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5,
    "truck": 7, "boat": 8, "traffic_light": 9, "cat": 15, "dog": 16,
    "bottle": 39, "chair": 56,
}

# Per-dataset: dataset's own class name -> COCO class name.
DATASET_CLASSES = {
    "pascalraw": {"person": "person", "bicycle": "bicycle", "car": "car"},
    "nod":       {"person": "person", "bicycle": "bicycle", "car": "car"},
    # ---- placeholders: replace with each dataset's real class strings ----
    "lod":       {"bicycle": "bicycle", "boat": "boat", "bottle": "bottle",
                  "bus": "bus", "car": "car", "cat": "cat", "chair": "chair",
                  "motorbike": "motorcycle"},
    "rod":       {"car": "car", "person": "person", "bus": "bus",
                  "truck": "truck", "bicycle": "bicycle"},
    "aodraw":    {"person": "person", "car": "car", "bicycle": "bicycle",
                  "bus": "bus", "truck": "truck"},   # COCO-overlap subset only
}


def dataset_to_coco_labels(dataset: str) -> dict:
    """Map a dataset's own class names straight to COCO contiguous labels."""
    return {cls: COCO_LABEL[coco] for cls, coco in DATASET_CLASSES[dataset].items()}


def evaluable_classes(datasets) -> list:
    """COCO class names common to ALL given datasets -- the only classes on which
    a frozen detector can be scored fairly across this fold."""
    sets = [set(DATASET_CLASSES[d].values()) for d in datasets]
    common = set.intersection(*sets)
    return sorted(common, key=lambda c: COCO_LABEL[c])


def remap_labels(dataset: str, names, keep):
    """Convert a list of a dataset's class names to COCO labels, returning None
    for any class not in `keep` (so the caller drops those boxes)."""
    d2c = DATASET_CLASSES[dataset]
    keep = set(keep)
    return [COCO_LABEL[d2c[n]] if (n in d2c and d2c[n] in keep) else None
            for n in names]


if __name__ == "__main__":
    assert dataset_to_coco_labels("pascalraw") == {"person": 0, "bicycle": 1, "car": 2}

    all5 = evaluable_classes(["pascalraw", "nod", "lod", "rod", "aodraw"])
    pn = evaluable_classes(["pascalraw", "nod"])
    print("evaluable across all 5 :", all5)            # intersection of real class sets
    print("evaluable pascalraw+nod:", pn)

    r = remap_labels("lod", ["car", "cat", "bicycle"], keep=["car", "bicycle"])
    assert r == [2, None, 1], r                        # 'cat' dropped (not kept)
    print("remap lod [car,cat,bicycle] keep{car,bicycle} ->", r)
    print("TAXONOMY OK")
