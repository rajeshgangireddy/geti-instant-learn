#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')
logger = logging.getLogger('selective_ptq')


def preprocess_image(image_path: Path, target_size: int = 1008) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert('RGB').resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[np.newaxis]


def build_calibration_data(calibration_dir: Path, num_calib: int) -> list[dict[str, np.ndarray]]:
    image_paths = sorted(calibration_dir.glob('*.jpg'))[:num_calib]
    if not image_paths:
        raise FileNotFoundError(f'No calibration images found in {calibration_dir}')
    cal_data = [{'pixel_values': preprocess_image(path)} for path in image_paths]
    logger.info('Calibration samples: %d', len(cal_data))
    return cal_data


def make_ignored_scope(nncf, model, variant: str):
    available_types = {op.get_type_name() for op in model.get_ops()}

    if variant == 'softmax_ln_gelu_add':
        requested = ['Softmax', 'MVN', 'Gelu', 'Add']
    elif variant == 'softmax_ln_gelu':
        requested = ['Softmax', 'MVN', 'Gelu']
    elif variant == 'softmax_only':
        requested = ['Softmax']
    elif variant == 'none':
        return None
    else:
        raise ValueError(f'Unknown ignored scope variant: {variant}')

    matched = [op_type for op_type in requested if op_type in available_types]
    logger.info('Ignoring op types for %s: %s', variant, matched)
    return nncf.IgnoredScope(types=matched)


def quantize_variant(core, nncf, vision_xml: Path, cal_data, output_dir: Path, target_device: str, preset: str, ignored_variant: str) -> Path:
    device_map = {'ANY': nncf.TargetDevice.ANY, 'GPU': nncf.TargetDevice.GPU}
    preset_map = {'PERFORMANCE': nncf.QuantizationPreset.PERFORMANCE, 'MIXED': nncf.QuantizationPreset.MIXED}

    model = core.read_model(vision_xml)
    ignored_scope = make_ignored_scope(nncf, model, ignored_variant)
    q_model = nncf.quantize(
        model,
        nncf.Dataset(cal_data),
        target_device=device_map[target_device],
        preset=preset_map[preset],
        subset_size=len(cal_data),
        model_type=nncf.ModelType.TRANSFORMER,
        ignored_scope=ignored_scope,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_xml = output_dir / 'vision-encoder.xml'
    import openvino as ov

    ov.save_model(q_model, str(out_xml))
    meta = {
        'target_device': target_device,
        'preset': preset,
        'ignored_scope': ignored_variant,
        'subset_size': len(cal_data),
        'source_model': str(vision_xml),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info('Saved %s', out_xml)
    return out_xml


def main() -> int:
    parser = argparse.ArgumentParser(description='Selective W8A8 PTQ for SAM3 vision encoder')
    parser.add_argument('--fp16-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--calib-dir', type=Path, required=True)
    parser.add_argument('--num-calib', type=int, default=100)
    parser.add_argument('--variants', nargs='*', default=[
        'ptq_ve_gpu_performance_softmax_ln_gelu',
        'ptq_ve_any_performance_softmax_ln_gelu',
        'ptq_ve_gpu_performance_softmax_only',
        'ptq_ve_gpu_performance_none',
    ])
    args = parser.parse_args()

    import openvino as ov
    import nncf

    core = ov.Core()
    vision_xml = args.fp16_dir / 'vision-encoder.xml'
    cal_data = build_calibration_data(args.calib_dir, args.num_calib)

    variant_map = {
        'ptq_ve_gpu_performance_softmax_ln_gelu_add': ('GPU', 'PERFORMANCE', 'softmax_ln_gelu_add'),
        'ptq_ve_gpu_performance_softmax_ln_gelu': ('GPU', 'PERFORMANCE', 'softmax_ln_gelu'),
        'ptq_ve_any_performance_softmax_ln_gelu': ('ANY', 'PERFORMANCE', 'softmax_ln_gelu'),
        'ptq_ve_gpu_performance_softmax_only': ('GPU', 'PERFORMANCE', 'softmax_only'),
        'ptq_ve_gpu_performance_none': ('GPU', 'PERFORMANCE', 'none'),
    }

    for variant in args.variants:
        if variant not in variant_map:
            raise ValueError(f'Unknown variant {variant}')
        target_device, preset, ignored_variant = variant_map[variant]
        out_dir = args.output_root / variant
        out_xml = out_dir / 'vision-encoder.xml'
        if out_xml.exists():
            logger.info('Already exists: %s', out_xml)
            continue
        logger.info('Quantizing %s', variant)
        quantize_variant(core, nncf, vision_xml, cal_data, out_dir, target_device, preset, ignored_variant)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
