"""Tests for COCO labels data structures and utilities."""

import json
from pathlib import Path

import pytest
import torch
from pycocotools.coco import COCO

from cuvis_ai_core.data.coco_labels import (
    Annotation,
    Category,
    COCOData,
    Image,
    Info,
    License,
    QueryableList,
    RLE2mask,
)


@pytest.fixture
def minimal_coco_json():
    """Create a minimal COCO format JSON for testing."""
    return {
        "info": {
            "description": "Test Dataset",
            "url": "https://test.com",
            "version": 1,
            "contributor": "Test User",
            "date_created": "2026-02-09",
        },
        "licenses": [
            {"id": 1, "name": "Test License", "url": "https://test.com/license"}
        ],
        "images": [
            {
                "id": 1,
                "file_name": "test_image_1.jpg",
                "width": 640,
                "height": 480,
                "license": 1,
            },
            {
                "id": 2,
                "file_name": "test_image_2.jpg",
                "width": 800,
                "height": 600,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "area": 2500.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[10, 10, 10, 20, 20, 20, 20, 10]],
                "area": 100.0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }


@pytest.fixture
def coco_json_file(minimal_coco_json, tmp_path):
    """Create a temporary COCO JSON file."""
    json_path = tmp_path / "test_coco.json"
    with open(json_path, "w") as f:
        json.dump(minimal_coco_json, f)
    return json_path


@pytest.fixture
def coco_data(coco_json_file):
    """Create a COCOData instance from the test file."""
    return COCOData.from_path(coco_json_file)


class TestCOCODataInitialization:
    """Test COCOData initialization and type annotations."""

    def test_init_type_annotations(self, coco_json_file):
        """Test that __init__ properly initializes typed attributes."""
        coco_data = COCOData.from_path(coco_json_file)

        # Check that type-annotated attributes are initialized to None
        assert coco_data._image_ids is None
        assert coco_data._categories is None
        assert coco_data._category_id_to_name is None
        assert coco_data._annotations is None
        assert coco_data._images is None

    def test_from_path_classmethod(self, coco_json_file):
        """Test from_path classmethod creates COCOData instance."""
        coco_data = COCOData.from_path(coco_json_file)
        assert isinstance(coco_data, COCOData)
        assert coco_data._coco is not None
        assert isinstance(coco_data._coco, COCO)


class TestCOCODataProperties:
    """Test COCOData lazy-loaded properties."""

    def test_image_ids_property(self, coco_data):
        """Test image_ids property returns sorted list."""
        image_ids = coco_data.image_ids
        assert isinstance(image_ids, list)
        assert len(image_ids) == 2
        assert image_ids == [1, 2]
        assert image_ids == sorted(image_ids)

    def test_info_property(self, coco_data):
        """Test info property returns Info dataclass."""
        info = coco_data.info
        assert isinstance(info, Info)
        assert info.description == "Test Dataset"
        assert info.version == 1

    def test_license_property(self, coco_data):
        """Test license property returns License (not Info)."""
        # This tests the bug fix where it was returning Info.from_dict
        license_obj = coco_data.license
        assert isinstance(license_obj, License)
        assert license_obj.id == 1
        assert license_obj.name == "Test License"

    def test_categories_property(self, coco_data):
        """Test categories property returns list of Category objects."""
        categories = coco_data.categories
        assert isinstance(categories, list)
        assert len(categories) == 2
        assert all(isinstance(cat, Category) for cat in categories)
        assert categories[0].name in ["person", "car"]

    def test_category_id_to_name_property(self, coco_data):
        """Test category_id_to_name property returns correct mapping."""
        mapping = coco_data.category_id_to_name
        assert isinstance(mapping, dict)
        assert mapping[1] in ["person", "car"]
        assert mapping[2] in ["person", "car"]
        assert len(mapping) == 2

    def test_images_property(self, coco_data):
        """Test images property returns list of Image objects."""
        images = coco_data.images
        assert isinstance(images, list)
        assert len(images) == 2
        assert all(isinstance(img, Image) for img in images)

    def test_annotations_property(self, coco_data):
        """Test annotations property returns QueryableList."""
        annotations = coco_data.annotations
        assert isinstance(annotations, QueryableList)
        assert len(annotations) == 2


class TestCOCODataSave:
    """Test COCOData save method with new type-safe implementation."""

    def test_save_creates_file(self, coco_data, tmp_path):
        """Test that save creates a valid JSON file."""
        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_save_preserves_annotations(self, coco_data, tmp_path):
        """Test that save preserves all annotations correctly."""
        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["annotations"]) == 2
        assert data["annotations"][0]["id"] == 1
        assert data["annotations"][1]["id"] == 2

    def test_save_handles_annotation_objects(self, coco_data, tmp_path):
        """Test save handles Annotation dataclass instances."""
        # Access annotations to ensure they're loaded as Annotation objects
        annotations = coco_data.annotations
        assert len(annotations) > 0
        assert isinstance(annotations[0], Annotation)

        output_path = tmp_path / "output_coco.json"
        coco_data.save(output_path)

        with open(output_path) as f:
            data = json.load(f)

        # Verify annotations were properly serialized
        assert isinstance(data["annotations"], list)
        assert len(data["annotations"]) == 2

    def test_save_with_path_string(self, coco_data, tmp_path):
        """Test save accepts string path."""
        output_path = str(tmp_path / "output_string.json")
        coco_data.save(output_path)
        assert Path(output_path).exists()

    def test_save_with_path_object(self, coco_data, tmp_path):
        """Test save accepts Path object."""
        output_path = tmp_path / "output_path.json"
        coco_data.save(output_path)
        assert output_path.exists()


class TestQueryableList:
    """Test QueryableList filtering functionality."""

    def test_where_filters_by_single_condition(self, coco_data):
        """Test where method filters items by single attribute."""
        annotations = coco_data.annotations
        filtered = annotations.where(image_id=1)

        assert len(filtered) == 2
        assert all(ann.image_id == 1 for ann in filtered)

    def test_where_filters_by_multiple_conditions(self, coco_data):
        """Test where method filters by multiple attributes."""
        annotations = coco_data.annotations
        filtered = annotations.where(image_id=1, category_id=1)

        assert len(filtered) == 1
        assert filtered[0].image_id == 1
        assert filtered[0].category_id == 1

    def test_queryable_list_iteration(self, coco_data):
        """Test QueryableList is iterable."""
        annotations = coco_data.annotations
        items = list(annotations)
        assert len(items) == 2

    def test_queryable_list_len(self, coco_data):
        """Test QueryableList supports len()."""
        annotations = coco_data.annotations
        assert len(annotations) == 2

    def test_queryable_list_indexing(self, coco_data):
        """Test QueryableList supports indexing."""
        annotations = coco_data.annotations
        first = annotations[0]
        assert isinstance(first, Annotation)


class TestAnnotation:
    """Test Annotation dataclass and torchvision conversion."""

    def test_annotation_to_dict_safe(self):
        """Test Annotation.to_dict_safe() preserves non-serializable objects."""
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=1,
            bbox=[0, 0, 10, 10],
            area=100.0,
        )
        result = ann.to_dict_safe()
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["bbox"] == [0, 0, 10, 10]

    def test_annotation_to_torchvision_bbox(self):
        """Test to_torchvision converts bbox to BoundingBoxes."""
        ann = Annotation(
            id=1,
            image_id=1,
            category_id=1,
            bbox=[100.0, 100.0, 50.0, 50.0],
            area=2500.0,
        )
        result = ann.to_torchvision(size=(480, 640))

        assert "bbox" in result
        bbox = result["bbox"]
        assert isinstance(bbox, torch.Tensor)
        assert bbox.shape == (1, 4)

    def test_annotation_to_torchvision_segmentation(self):
        """Test to_torchvision converts polygon segmentation to Mask."""
        ann = Annotation(
            id=2,
            image_id=1,
            category_id=2,
            segmentation=[[10, 10, 10, 20, 20, 20, 20, 10]],
            area=100.0,
        )
        result = ann.to_torchvision(size=(480, 640))

        assert "segmentation" in result

    def test_annotation_to_torchvision_rle_mask(self):
        """Test to_torchvision converts RLE mask."""
        ann = Annotation(
            id=3,
            image_id=1,
            category_id=1,
            mask={"size": [10, 10], "counts": [5, 2, 3, 10, 80]},
        )
        result = ann.to_torchvision(size=(10, 10))

        assert "mask" in result


class TestRLE2mask:
    """Test RLE to mask conversion utility."""

    def test_rle2mask_basic(self):
        """Test RLE2mask converts run-length encoding to binary mask."""
        rle = [5, 2, 3, 10, 80]  # 5 zeros, 2 ones, 3 zeros, 10 ones, 80 zeros
        mask = RLE2mask(rle, mask_width=10, mask_height=10)

        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        assert mask.sum() == 12  # 2 + 10 ones

    def test_rle2mask_all_zeros(self):
        """Test RLE2mask with all zeros."""
        rle = [100]
        mask = RLE2mask(rle, mask_width=10, mask_height=10)

        assert mask.shape == (10, 10)
        assert mask.sum() == 0

    def test_rle2mask_alternating(self):
        """Test RLE2mask with alternating pattern."""
        rle = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        mask = RLE2mask(rle, mask_width=5, mask_height=2)

        assert mask.shape == (2, 5)
        assert mask.sum() == 5
