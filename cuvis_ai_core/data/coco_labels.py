import contextlib
import json
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataclass_wizard import JSONWizard
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from torchvision.tv_tensors import BoundingBoxes, Mask

import io

# from contextlib import contextmanager


def RLE2mask(rle: list, mask_width: int, mask_height: int) -> np.ndarray:
    mask = np.zeros(mask_width * mask_height, np.uint8)
    ids = 0
    value = 0
    for c in rle:
        mask[ids : ids + c] = value
        value = not value
        ids += c
    mask = mask.reshape((mask_height, mask_width), order="F")
    return mask.astype(bool, copy=False)


class SafeWizard(JSONWizard):
    """
    JSONWizard subclass that safely converts dataclasses to dicts,
    keeping non-serializable objects (e.g., torch Tensors, Masks)
    as-is instead of falling back to string representations.
    """

    def to_dict_safe(self) -> dict[str, Any]:
        """
        Like `to_dict()`, but leaves unsupported types untouched.
        """
        base_dict = super().to_dict()
        final_dict = {}

        for key, value in vars(self).items():
            if not self._is_json_serializable(value):
                # keep original object (Mask, Tensor, etc.)
                final_dict[key] = value
                continue
            val = base_dict.get(key, value)
            final_dict[key] = val
        return final_dict

    @staticmethod
    def _is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except Exception:
            return False


@dataclass
class Info(JSONWizard):
    description: str | None = None
    url: str | None = None
    version: int | None = None
    contributor: str | None = None
    date_created: str | None = None


@dataclass
class License(JSONWizard):
    id: int
    name: str
    url: str | None = None


@dataclass
class Category(JSONWizard):
    id: int
    name: str
    supercategory: str | None = None


@dataclass
class Image(JSONWizard):
    id: int
    file_name: str
    height: int
    width: int
    license: int | None = None
    flickr_url: str | None = None
    coco_url: str | None = None
    date_captured: str | None = None
    wavelength: list[float] | None = field(default_factory=list)


@dataclass
class Annotation(SafeWizard):
    id: int
    image_id: int
    category_id: int
    segmentation: list | None = None
    area: float | None = None
    bbox: list[float] | None = None
    mask: dict | None = None
    iscrowd: int | None = 0
    auxiliary: dict[str, Any] | None = field(default_factory=dict)

    def to_dict_safe(self) -> dict[str, Any]:
        """
        Like `to_dict()`, but leaves unsupported types untouched.
        """
        base_dict = super().to_dict()
        final_dict = {}

        for key, value in vars(self).items():
            if not self._is_json_serializable(value):
                # keep original object (Mask, Tensor, etc.)
                final_dict[key] = value
                continue
            val = base_dict.get(key, value)
            final_dict[key] = val
        return final_dict

    @staticmethod
    def _is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except Exception:
            return False

    def to_torchvision(self, size: tuple[int, int]) -> dict[str, Any]:
        """Convert COCO-style bbox/segmentation/mask into torchvision tensors."""
        out = copy(self)

        if self.bbox is not None:
            out.bbox = BoundingBoxes(
                torch.tensor([self.bbox], dtype=torch.float32),
                format="XYWH",
                canvas_size=size,
            )

        if (
            self.segmentation is not None
            and isinstance(self.segmentation, list)
            and self.segmentation != []
        ):
            coords = np.array(self.segmentation[0]).reshape(-1, 2)
            mask_np = polygon2mask(size, coords).astype(np.uint8)
            out.segmentation = Mask(torch.from_numpy(mask_np))

        if self.mask is not None:
            size = self.mask["size"]
            mask_np = RLE2mask(self.mask["counts"], size[0], size[1])
            out.mask = Mask(torch.from_numpy(mask_np))

        return out.to_dict_safe()
        # return out


class QueryableList:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def where(self, **conditions) -> list[Any]:
        """
        Filter items based on conditions.
        :param conditions: Keyword arguments representing field=value filters.
        :return: A new QueryableList with filtered items.
        """
        filtered_items = self._items
        for key, value in conditions.items():
            filtered_items = [
                item for item in filtered_items if getattr(item, key) == value
            ]
        return list(filtered_items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]


class COCOData:
    def __init__(self, coco: COCO) -> None:
        self._coco = coco
        self._image_ids: list[int] | None = None
        self._categories: list[Category] | None = None
        self._category_id_to_name: dict[int, str] | None = None
        self._annotations: QueryableList | None = None
        self._images: list[Image] | None = None

    @classmethod
    def from_path(cls, path: Path | str):
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(COCO(path))

    # @classmethod
    # def from_path(cls, path):
    #     old_print = builtins.print
    #     builtins.print = lambda *a, **k: None
    #     try:
    #         return cls(COCO(str(path)))
    #     finally:
    #         builtins.print = old_print

    @property
    def image_ids(self) -> list[int]:
        if self._image_ids is None:
            self._image_ids = sorted(self._coco.imgs.keys())
        return self._image_ids

    @property
    def info(self) -> Info:
        return Info.from_dict(self._coco.dataset["info"])

    @property
    def license(self) -> License:
        return License.from_dict(self._coco.dataset["licenses"][0])

    @property
    def annotations(self) -> QueryableList:
        if self._annotations is None:
            self._annotations = QueryableList(
                [Annotation.from_dict(v) for v in self._coco.anns.values()]
            )
        return self._annotations

    @property
    def categories(self) -> list[Category]:
        if self._categories is None:
            self._categories = [Category.from_dict(v) for v in self._coco.cats.values()]
        return self._categories

    @property
    def category_id_to_name(self) -> dict[int, str]:
        if self._category_id_to_name is None:
            self._category_id_to_name = {cat.id: cat.name for cat in self.categories}
        return self._category_id_to_name

    @property
    def images(self) -> list[Image]:
        if self._images is None:
            self._images = [Image.from_dict(v) for v in self._coco.imgs.values()]
        return self._images

    def save(self, path: str | Path) -> None:
        """
        Save the current COCOData object (images, annotations, categories, etc.)
        back into a COCO-style JSON file.

        Automatically converts dataclasses to plain dicts and ensures
        compliance with standard COCO structure.
        """
        path = str(path)
        annotations_list: list[dict[str, Any]] = []

        ann: Annotation | dict[str, Any]
        for ann in self.annotations:
            if isinstance(ann, Annotation):
                annotations_list.append(ann.to_dict_safe())
            elif isinstance(ann, dict):
                annotations_list.append(ann)
            else:
                raise TypeError(f"Unsupported annotation type: {type(ann)}")

        dataset = {
            "info": self.info.to_dict() if hasattr(self, "info") else {},
            "licenses": self._coco.dataset.get("licenses", []),
            "images": [img.to_dict() for img in self.images],
            "annotations": annotations_list,
            "categories": [cat.to_dict() for cat in self.categories],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        print(f"COCOData saved successfully to: {path}")


if __name__ == "__main__":
    from pycocotools.coco import COCO
    import contextlib
    import io

    ann_file = r"..\data\Lentils\Lentils_000.json"

    # with contextlib.redirect_stdout(io.StringIO()):

    # coco = COCO(ann_file)
    coco_data = COCOData.from_path(ann_file)
