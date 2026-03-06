# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models including PerDino, Matcher, SoftMatcher, GroundedSAM, and YOLOE."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision.tv_tensors import Image

from instantlearn.models.grounded_sam import GroundedSAM
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher
from instantlearn.models.yoloe import YOLOE


class TestPerDino:
    """Test PerDino model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for PerDino."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "similarity_matcher": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino initialization."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "similarity_matcher")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino forward pass."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        # Mock the fit method to set up the model
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that PerDino uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device="cpu")

        # Mock the fit and predict methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestMatcher:
    """Test Matcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher initialization."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher forward pass."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that Matcher uses multi-instance prior filtering."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestSoftMatcher:
    """Test SoftMatcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher initialization with new components."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher forward pass with new architecture."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_multi_instance_filtering(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that SoftMatcher uses multi-instance prior filtering."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for GroundedSAM."""
        return {
            "sam_predictor": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
            "multi_instance_prior_filter": MagicMock(),
        }

    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "prompt_filter")

    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_forward_pass(self, mock_load_sam: MagicMock, mock_components: dict[str, Any]) -> None:
        """Test GroundedSAM forward pass with new architecture."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        # Create test data
        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)

    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_multi_instance_filtering(
        self,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test that GroundedSAM uses multi-instance prior filtering."""
        mock_load_sam.return_value = mock_components["sam_predictor"]

        model = GroundedSAM(device="cpu")

        # Mock the learn and infer methods
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_points": torch.zeros((0, 4), dtype=torch.float32),
                    "pred_boxes": torch.zeros((0, 6), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        model.predict(target_images)

        # Verify that predict was called
        model.predict.assert_called_once_with(target_images)


class TestYOLOE:
    """Test YOLOE model."""

    @patch("ultralytics.YOLO")
    def test_yoloe_initialization(self, mock_yolo_cls: MagicMock) -> None:
        """Test YOLOE initialization."""
        mock_yolo_cls.return_value = MagicMock()

        model = YOLOE(model_name="yoloe-v8s-seg", device="cpu")

        assert model.model_name == "yoloe-v8s-seg"
        assert model.confidence_threshold == 0.25
        assert model.iou_threshold == 0.7
        assert model.imgsz == 640
        assert model.use_nms is True
        assert model.precision == "fp16"
        mock_yolo_cls.assert_called_once()

    def test_yoloe_invalid_model_name(self) -> None:
        """Test YOLOE raises ValueError for unknown model name."""
        with pytest.raises(ValueError, match="Unknown YOLOE model"):
            YOLOE(model_name="yoloe-invalid", device="cpu")

    @patch("ultralytics.YOLO")
    def test_yoloe_fit_sets_visual_prompts(self, mock_yolo_cls: MagicMock) -> None:
        """Test that fit() stores visual prompts and reference image."""
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        model = YOLOE(model_name="yoloe-v8s-seg", device="cpu")

        # Create a mock reference sample
        ref_image = torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)
        ref_mask = torch.zeros(100, 100, dtype=torch.bool)
        ref_mask[20:50, 30:60] = True

        with patch("instantlearn.models.yoloe.yoloe.Batch") as mock_batch_cls:
            mock_batch = MagicMock()
            mock_batch.images = [ref_image]
            mock_batch.masks = [ref_mask.unsqueeze(0)]
            mock_batch.categories = [["object"]]
            mock_batch_cls.collate.return_value = mock_batch

            model.fit(MagicMock())

        assert model._visual_prompts_set is True
        assert model._refer_image is not None
        assert model._visual_prompts["bboxes"] == [[30, 20, 59, 49]]
        assert model._visual_prompts["cls"] == ["object"]

    @patch("ultralytics.YOLO")
    def test_yoloe_predict_raises_without_fit(self, mock_yolo_cls: MagicMock) -> None:
        """Test that predict() raises RuntimeError if fit() was not called."""
        mock_yolo_cls.return_value = MagicMock()

        model = YOLOE(model_name="yoloe-v8s-seg", device="cpu")

        with pytest.raises(RuntimeError, match="No visual prompts set"):
            model.predict([Image(torch.zeros((3, 224, 224), dtype=torch.uint8))])

    @patch("ultralytics.YOLO")
    def test_yoloe_forward_pass(self, mock_yolo_cls: MagicMock) -> None:
        """Test YOLOE forward pass with mocked methods."""
        mock_yolo_cls.return_value = MagicMock()

        model = YOLOE(model_name="yoloe-v8s-seg", device="cpu")

        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(
            return_value=[
                {
                    "pred_masks": torch.zeros((0, 224, 224), dtype=torch.bool),
                    "pred_boxes": torch.zeros((0, 5), dtype=torch.float32),
                    "pred_labels": torch.zeros((0,), dtype=torch.long),
                },
            ],
        )

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]

        predictions = model.predict(target_images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]
        model.predict.assert_called_once_with(target_images)
