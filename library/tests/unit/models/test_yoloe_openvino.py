# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for YOLOE OpenVINO model and postprocessing utilities."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image

from instantlearn.models.yoloe.postprocessing import (
    letterbox_image,
    parse_detections,
    preprocess_image,
    process_mask_protos,
    scale_boxes_to_original,
)


class TestLetterboxImage:
    """Test letterbox_image utility."""

    def test_square_image_no_padding(self) -> None:
        """Test that a square image matching target needs no padding."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, pad = letterbox_image(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert scale == 1.0
        assert pad == (0, 0)

    def test_landscape_image_gets_top_bottom_padding(self) -> None:
        """Test that a wide image gets vertical padding."""
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        padded, scale, pad = letterbox_image(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 600, abs=0.01)
        assert pad[0] == 0  # no horizontal padding
        assert pad[1] > 0  # vertical padding

    def test_portrait_image_gets_left_right_padding(self) -> None:
        """Test that a tall image gets horizontal padding."""
        img = np.zeros((600, 300, 3), dtype=np.uint8)
        padded, scale, pad = letterbox_image(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert pad[0] > 0  # horizontal padding
        assert pad[1] == 0  # no vertical padding

    def test_fill_color(self) -> None:
        """Test that padding uses the specified fill color."""
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        padded, _, _ = letterbox_image(img, (640, 640), color=(0, 0, 0))
        # Corners should be padding color (0, 0, 0)
        assert np.all(padded[0, 0] == [0, 0, 0])


class TestPreprocessImage:
    """Test preprocess_image utility."""

    def test_output_shape_and_dtype(self) -> None:
        """Test that output is NCHW float32 in [0, 1]."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob, scale, pad = preprocess_image(img, (640, 640))
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0


class TestParseDetections:
    """Test parse_detections utility."""

    def test_filters_by_confidence(self) -> None:
        """Test that low-confidence detections are filtered out."""
        # 3 detections, confidence = [0.1, 0.5, 0.8]
        dets = np.zeros((1, 3, 38), dtype=np.float32)
        dets[0, 0, 4] = 0.1  # below threshold
        dets[0, 1, 4] = 0.5
        dets[0, 2, 4] = 0.8
        # Set some box values
        dets[0, 1, :4] = [10, 20, 100, 200]
        dets[0, 2, :4] = [50, 60, 150, 250]
        dets[0, 1, 5] = 0  # class 0
        dets[0, 2, 5] = 1  # class 1

        boxes, scores, class_ids, mask_coeffs = parse_detections(dets, confidence_threshold=0.25)

        assert len(boxes) == 2
        assert len(scores) == 2
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(0.8)
        assert class_ids[0] == 0
        assert class_ids[1] == 1
        assert mask_coeffs.shape == (2, 32)

    def test_all_below_threshold_returns_empty(self) -> None:
        """Test that no detections pass when all are below threshold."""
        dets = np.zeros((1, 5, 38), dtype=np.float32)
        dets[0, :, 4] = 0.01  # all low confidence

        boxes, scores, class_ids, mask_coeffs = parse_detections(dets, confidence_threshold=0.25)

        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(class_ids) == 0
        assert mask_coeffs.shape == (0, 32)

    def test_custom_nm(self) -> None:
        """Test with non-default number of mask coefficients."""
        nm = 16
        dets = np.zeros((1, 2, 6 + nm), dtype=np.float32)
        dets[0, 0, 4] = 0.9
        dets[0, 1, 4] = 0.9

        _, _, _, mask_coeffs = parse_detections(dets, confidence_threshold=0.5, nm=nm)

        assert mask_coeffs.shape == (2, nm)


class TestScaleBoxesToOriginal:
    """Test scale_boxes_to_original utility."""

    def test_identity_when_no_padding_or_scale(self) -> None:
        """Test that boxes stay the same with scale=1 and no padding."""
        boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        result = scale_boxes_to_original(boxes, scale=1.0, pad=(0, 0), original_shape=(640, 640))
        np.testing.assert_array_almost_equal(result, boxes)

    def test_removes_padding_and_rescales(self) -> None:
        """Test that padding is removed and boxes are rescaled."""
        # Box at (100, 100, 200, 200) in letterbox space
        # With pad=(50, 0) and scale=0.5
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        result = scale_boxes_to_original(
            boxes, scale=0.5, pad=(50, 0), original_shape=(1000, 1000)
        )
        # After removing pad: (50, 100, 150, 200), then /0.5: (100, 200, 300, 400)
        expected = np.array([[100, 200, 300, 400]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_boxes(self) -> None:
        """Test with empty input."""
        boxes = np.empty((0, 4), dtype=np.float32)
        result = scale_boxes_to_original(boxes, scale=0.5, pad=(10, 10), original_shape=(480, 640))
        assert result.shape == (0, 4)

    def test_clips_to_image_bounds(self) -> None:
        """Test that boxes are clipped to valid coordinates."""
        boxes = np.array([[-50, -50, 700, 700]], dtype=np.float32)
        result = scale_boxes_to_original(
            boxes, scale=1.0, pad=(0, 0), original_shape=(640, 640)
        )
        assert result[0, 0] == 0  # clipped x1
        assert result[0, 1] == 0  # clipped y1
        assert result[0, 2] == 640  # clipped x2
        assert result[0, 3] == 640  # clipped y2


class TestProcessMaskProtos:
    """Test process_mask_protos utility."""

    def test_empty_coefficients_returns_empty(self) -> None:
        """Test with no detections."""
        mask_coeffs = np.empty((0, 32), dtype=np.float32)
        protos = np.random.randn(1, 32, 160, 160).astype(np.float32)
        boxes = np.empty((0, 4), dtype=np.float32)

        masks = process_mask_protos(
            mask_coeffs, protos, boxes,
            original_shape=(480, 640), scale=1.0, pad=(0, 0),
        )
        assert masks.shape == (0, 480, 640)

    def test_output_shape_matches_original(self) -> None:
        """Test that output masks match original image dimensions."""
        np.random.seed(42)
        mask_coeffs = np.random.randn(2, 32).astype(np.float32)
        protos = np.random.randn(1, 32, 160, 160).astype(np.float32)
        boxes = np.array([
            [100, 100, 300, 300],
            [200, 200, 400, 400],
        ], dtype=np.float32)

        masks = process_mask_protos(
            mask_coeffs, protos, boxes,
            original_shape=(480, 640), scale=1.0, pad=(0, 0),
        )
        assert masks.shape == (2, 480, 640)
        assert masks.dtype == bool


class TestYOLOEOpenVINO:
    """Test YOLOEOpenVINO model class."""

    @pytest.fixture
    def mock_ov_model(self) -> dict[str, Any]:
        """Create mock OpenVINO model components."""
        mock_infer_request = MagicMock()

        # Mock output tensors
        # det_output: [1, 300, 38]
        det_data = np.zeros((1, 300, 38), dtype=np.float32)
        det_data[0, 0, :4] = [100, 100, 200, 200]  # box
        det_data[0, 0, 4] = 0.9  # score
        det_data[0, 0, 5] = 0  # class
        det_data[0, 1, :4] = [300, 300, 400, 400]
        det_data[0, 1, 4] = 0.8
        det_data[0, 1, 5] = 1

        proto_data = np.random.randn(1, 32, 160, 160).astype(np.float32)

        det_tensor = MagicMock()
        det_tensor.data = det_data
        proto_tensor = MagicMock()
        proto_tensor.data = proto_data

        mock_infer_request.get_output_tensor.side_effect = lambda idx: (
            det_tensor if idx == 0 else proto_tensor
        )

        # Mock compiled model
        mock_compiled = MagicMock()
        mock_compiled.create_infer_request.return_value = mock_infer_request

        # Mock output shape
        mock_partial_shape = MagicMock()
        mock_partial_shape.__len__ = lambda _: 3
        dim_300 = MagicMock()
        dim_300.get_length.return_value = 300
        dim_38 = MagicMock()
        dim_38.get_length.return_value = 38
        dim_1 = MagicMock()
        dim_1.get_length.return_value = 1
        mock_partial_shape.__getitem__ = lambda _, idx: [dim_1, dim_300, dim_38][idx]
        mock_output = MagicMock()
        mock_output.partial_shape = mock_partial_shape
        mock_compiled.output.return_value = mock_output

        return {
            "compiled_model": mock_compiled,
            "infer_request": mock_infer_request,
        }

    @patch("builtins.open", mock_open(read_data="imgsz: [640, 640]\nnames:\n  0: person\n  1: car\nstride: 32\ntask: segment\nend2end: true\n"))
    @patch("openvino.Core")
    def test_initialization(self, mock_ov_core: MagicMock, mock_ov_model: dict, tmp_path: Path) -> None:
        """Test YOLOEOpenVINO initialization."""
        # Create a dummy XML file
        xml_file = tmp_path / "model.xml"
        xml_file.touch()
        meta_file = tmp_path / "metadata.yaml"
        meta_file.touch()

        mock_core_instance = MagicMock()
        mock_ov_core.return_value = mock_core_instance
        mock_core_instance.compile_model.return_value = mock_ov_model["compiled_model"]
        mock_core_instance.read_model.return_value = MagicMock()

        from instantlearn.models.yoloe.yoloe_openvino import YOLOEOpenVINO

        model = YOLOEOpenVINO(model_dir=tmp_path, device="cpu")

        assert model.confidence_threshold == 0.25
        assert model.iou_threshold == 0.7
        assert model._nm == 32

    @patch("builtins.open", mock_open(read_data="imgsz: [640, 640]\nnames:\n  0: person\n  1: car\nstride: 32\n"))
    @patch("openvino.Core")
    def test_predict_returns_correct_format(
        self, mock_ov_core: MagicMock, mock_ov_model: dict, tmp_path: Path
    ) -> None:
        """Test that predict returns the expected dictionary format."""
        xml_file = tmp_path / "model.xml"
        xml_file.touch()
        meta_file = tmp_path / "metadata.yaml"
        meta_file.touch()

        mock_core_instance = MagicMock()
        mock_ov_core.return_value = mock_core_instance
        mock_core_instance.compile_model.return_value = mock_ov_model["compiled_model"]
        mock_core_instance.read_model.return_value = MagicMock()

        from instantlearn.models.yoloe.yoloe_openvino import YOLOEOpenVINO

        model = YOLOEOpenVINO(model_dir=tmp_path, device="cpu")

        # Create target image
        target_image = Image(torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8))

        with patch("instantlearn.models.yoloe.yoloe_openvino.Batch") as mock_batch_cls:
            mock_batch = MagicMock()
            mock_batch.images = [target_image]
            mock_batch_cls.collate.return_value = mock_batch

            results = model.predict(target_image)

        assert len(results) == 1
        pred = results[0]
        assert "pred_masks" in pred
        assert "pred_boxes" in pred
        assert "pred_labels" in pred
        assert pred["pred_boxes"].shape[1] == 5  # x1, y1, x2, y2, score
        assert pred["pred_labels"].dtype == torch.long

    @patch("builtins.open", mock_open(read_data="imgsz: [640, 640]\nnames:\n  0: person\nstride: 32\n"))
    @patch("openvino.Core")
    def test_fit_records_category_ids(
        self, mock_ov_core: MagicMock, mock_ov_model: dict, tmp_path: Path
    ) -> None:
        """Test that fit() records category ID mapping."""
        xml_file = tmp_path / "model.xml"
        xml_file.touch()
        meta_file = tmp_path / "metadata.yaml"
        meta_file.touch()

        mock_core_instance = MagicMock()
        mock_ov_core.return_value = mock_core_instance
        mock_core_instance.compile_model.return_value = mock_ov_model["compiled_model"]
        mock_core_instance.read_model.return_value = MagicMock()

        from instantlearn.models.yoloe.yoloe_openvino import YOLOEOpenVINO

        model = YOLOEOpenVINO(model_dir=tmp_path, device="cpu")

        with patch("instantlearn.models.yoloe.yoloe_openvino.Batch") as mock_batch_cls:
            mock_batch = MagicMock()
            mock_batch.category_ids = [torch.tensor([0]), torch.tensor([1])]
            mock_batch_cls.collate.return_value = mock_batch

            model.fit(MagicMock())

        assert model._category_id_map is not None
        assert 0 in model._category_id_map
        assert 1 in model._category_id_map

    @patch("builtins.open", mock_open(read_data="imgsz: [640, 640]\nnames:\n  0: person\nstride: 32\n"))
    @patch("openvino.Core")
    def test_export_returns_model_dir(
        self, mock_ov_core: MagicMock, mock_ov_model: dict, tmp_path: Path
    ) -> None:
        """Test that export() returns the model directory path."""
        xml_file = tmp_path / "model.xml"
        xml_file.touch()
        meta_file = tmp_path / "metadata.yaml"
        meta_file.touch()

        mock_core_instance = MagicMock()
        mock_ov_core.return_value = mock_core_instance
        mock_core_instance.compile_model.return_value = mock_ov_model["compiled_model"]
        mock_core_instance.read_model.return_value = MagicMock()

        from instantlearn.models.yoloe.yoloe_openvino import YOLOEOpenVINO

        model = YOLOEOpenVINO(model_dir=tmp_path, device="cpu")

        result = model.export()
        assert result == tmp_path

    def test_no_xml_raises_error(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when no XML file exists."""
        from instantlearn.models.yoloe.yoloe_openvino import YOLOEOpenVINO

        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            YOLOEOpenVINO(model_dir=tmp_path, device="cpu")


class TestExportScript:
    """Test the export script function."""

    @patch("ultralytics.YOLO")
    def test_export_yoloe_openvino(self, mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
        """Test the export function creates files in output directory."""
        mock_model = MagicMock()
        mock_inner = MagicMock()
        mock_model.model = mock_inner
        mock_inner.get_text_pe.return_value = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Simulate export returning a temporary directory
        ov_temp = tmp_path / "model_openvino_model"
        ov_temp.mkdir()
        (ov_temp / "model.xml").touch()
        (ov_temp / "model.bin").touch()
        (ov_temp / "metadata.yaml").write_text("imgsz: [640, 640]\nnames:\n  0: person\n")
        mock_model.export.return_value = str(ov_temp)

        from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino

        output_dir = tmp_path / "output"
        result = export_yoloe_openvino(
            model_name="yoloe-26s-seg",
            classes=["person", "car"],
            output_dir=output_dir,
            imgsz=640,
            half=False,
        )

        assert result.exists()
        mock_inner.get_text_pe.assert_called_once_with(["person", "car"])
        mock_inner.set_classes.assert_called_once()

    def test_export_invalid_model_raises(self, tmp_path: Path) -> None:
        """Test that export raises ValueError for unknown model."""
        from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino

        with pytest.raises(ValueError, match="Unknown YOLOE model"):
            export_yoloe_openvino(
                model_name="yoloe-invalid",
                classes=["person"],
                output_dir=tmp_path,
            )
