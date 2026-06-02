#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = Path.home() / '.cache/huggingface/hub/models--rajeshgangireddy--SAM3_OpenVINO/snapshots/844e4143273676daa3402f69c866f8a71f65067f'
FP16_DIR = HF_CACHE / 'openvino-fp16'
ROUND1_INT8_DIR = ROOT / 'agent-results/sam3_int8_gpu_round2/int8_sym_g128'
INT4_DIR = ROOT / 'agent-results/sam3_int8_gpu_round2/int4_ve'
CSV_PATH = ROOT / 'agent-results/sam3_int8_gpu_round2/bench_results.csv'
ACCURACY_JSON = ROOT / 'agent-results/sam3_int8_gpu_round2/accuracy_results.json'

AVG_RE = re.compile(r'Average:\s+([0-9.]+) ms')
MED_RE = re.compile(r'Median:\s+([0-9.]+) ms')
FIRST_RE = re.compile(r'First inference took\s+([0-9.]+) ms')


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=True)
    return proc.stdout + proc.stderr


def clear_cache(model_dir: Path) -> None:
    cache_dir = model_dir / '.ov_cache'
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def parse_benchmark_output(text: str) -> dict[str, float]:
    avg_match = AVG_RE.search(text)
    med_match = MED_RE.search(text)
    first_match = FIRST_RE.search(text)
    if not avg_match or not med_match:
        raise RuntimeError(f'Could not parse benchmark_app output:\n{text}')
    result = {
        'average_ms': float(avg_match.group(1)),
        'median_ms': float(med_match.group(1)),
    }
    if first_match:
        result['first_inference_ms'] = float(first_match.group(1))
    return result


def benchmark(model_xml: Path, benchmark_app: Path, extra_props: dict[str, str] | None = None) -> tuple[dict[str, float], str]:
    clear_cache(model_xml.parent)
    cmd = [
        str(benchmark_app),
        '-m', str(model_xml),
        '-d', 'GPU',
        '-hint', 'latency',
        '-niter', '200',
        '-nireq', '1',
        '-shape', '[1,3,1008,1008]',
        '-api', 'sync',
    ]
    config_path = None
    if extra_props:
        config = {'GPU': {'PERFORMANCE_HINT': 'LATENCY', 'PERFORMANCE_HINT_NUM_REQUESTS': '1', 'PERF_COUNT': 'NO', **extra_props}}
        config_path = model_xml.parent / 'benchmark_config.json'
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        cmd.extend(['-load_config', str(config_path)])
    text = run(cmd)
    return parse_benchmark_output(text), text


def export_weight_only_int8() -> Path | None:
    import openvino as ov
    import nncf

    out_xml = ROUND1_INT8_DIR / 'vision-encoder.xml'
    if out_xml.exists():
        return out_xml
    ROUND1_INT8_DIR.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(FP16_DIR / 'vision-encoder.xml')
    try:
        model = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8_SYM,
            group_size=128,
            scale_estimation=True,
        )
    except Exception as exc:  # noqa: BLE001
        (ROUND1_INT8_DIR / 'export_error.txt').write_text(str(exc), encoding='utf-8')
        return None
    ov.save_model(model, str(out_xml))
    (ROUND1_INT8_DIR / 'metadata.json').write_text(json.dumps({
        'mode': 'INT8_SYM',
        'group_size': 128,
        'scale_estimation': True,
    }, indent=2), encoding='utf-8')
    return out_xml


def export_weight_only_int4() -> Path:
    import openvino as ov
    import nncf

    out_xml = INT4_DIR / 'vision-encoder.xml'
    if out_xml.exists():
        return out_xml
    INT4_DIR.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(FP16_DIR / 'vision-encoder.xml')
    model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=64,
        scale_estimation=True,
    )
    ov.save_model(model, str(out_xml))
    (INT4_DIR / 'metadata.json').write_text(json.dumps({
        'mode': 'INT4_SYM',
        'group_size': 64,
        'scale_estimation': True,
    }, indent=2), encoding='utf-8')
    return out_xml


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0:
        return 1.0
    return float(np.dot(a64, b64) / denom)


def preprocess_image(image_path: Path, target_size: int = 1008) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert('RGB').resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[np.newaxis]


def accuracy_check(fp16_xml: Path, candidate_xml: Path, num_images: int = 50) -> dict[str, float | bool]:
    import openvino as ov

    image_dir = Path('/home/rgangire/workspace/data/prompt/lvis/val2017')
    image_paths = sorted(image_dir.glob('*.jpg'))[:num_images]
    core = ov.Core()
    fp16_compiled = core.compile_model(core.read_model(fp16_xml), 'GPU', {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'})
    cand_compiled = core.compile_model(core.read_model(candidate_xml), 'GPU', {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'})

    sims = []
    for image_path in image_paths:
        data = preprocess_image(image_path)
        fp16_out = fp16_compiled([data])['fpn_feat_2']
        cand_out = cand_compiled([data])['fpn_feat_2']
        sims.append(cosine_similarity(fp16_out, cand_out))
    mean_sim = float(np.mean(sims))
    return {
        'mean_cosine_sim': mean_sim,
        'pass': mean_sim > 0.99,
        'num_images': len(sims),
    }


def query_gpu_capabilities() -> dict[str, object]:
    import openvino as ov

    core = ov.Core()
    result = {}
    for prop in ['OPTIMIZATION_CAPABILITIES', 'FULL_DEVICE_NAME', 'SUPPORTED_PROPERTIES']:
        try:
            result[prop] = core.get_property('GPU', prop)
        except Exception as exc:  # noqa: BLE001
            result[prop] = f'ERROR: {exc}'
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark SAM3 quantized vision encoder variants')
    parser.add_argument('--benchmark-app', type=Path, required=True)
    parser.add_argument('--nightly-benchmark-app', type=Path)
    parser.add_argument('--output-csv', type=Path, default=CSV_PATH)
    parser.add_argument('--accuracy-candidates', nargs='*', default=[])
    args = parser.parse_args()

    rows = []
    capabilities = query_gpu_capabilities()
    (ROOT / 'agent-results/sam3_int8_gpu_round2/gpu_capabilities.json').write_text(json.dumps(capabilities, indent=2, default=str), encoding='utf-8')

    fp16_xml = FP16_DIR / 'vision-encoder.xml'
    int8_xml = export_weight_only_int8()
    int4_xml = export_weight_only_int4()

    variants = [
        ('fp16_baseline', 'FP16 baseline', fp16_xml, args.benchmark_app, None),
    ]
    if int8_xml is not None and int8_xml.exists():
        variants.append(('int8_sym_g128', 'compress_weights(INT8_SYM, g=128, scale_est=True)', int8_xml, args.benchmark_app, None))

    round2_root = ROOT / 'agent-results/sam3_int8_gpu_round2'
    ptq_candidates = [
        ('ptq_ve_gpu_performance_softmax_ln_gelu', 'Selective PTQ GPU PERFORMANCE ignore Softmax/LN/GELU', round2_root / 'ptq_ve_gpu_performance_softmax_ln_gelu/vision-encoder.xml', args.benchmark_app, None),
        ('ptq_ve_any_performance_softmax_ln_gelu', 'Selective PTQ ANY PERFORMANCE ignore Softmax/LN/GELU', round2_root / 'ptq_ve_any_performance_softmax_ln_gelu/vision-encoder.xml', args.benchmark_app, None),
        ('ptq_ve_gpu_performance_softmax_only', 'Selective PTQ GPU PERFORMANCE ignore Softmax only', round2_root / 'ptq_ve_gpu_performance_softmax_only/vision-encoder.xml', args.benchmark_app, None),
        ('ptq_ve_gpu_performance_none', 'Selective PTQ GPU PERFORMANCE no ignored scope', round2_root / 'ptq_ve_gpu_performance_none/vision-encoder.xml', args.benchmark_app, None),
        ('int4_ve', 'compress_weights(INT4_SYM, g=64, scale_est=True)', int4_xml, args.benchmark_app, None),
        ('int8_host_queue_high', 'INT8 g128 with GPU_HOST_TASK_PRIORITY=HIGH,GPU_QUEUE_PRIORITY=HIGH', int8_xml, args.benchmark_app, {'GPU_HOST_TASK_PRIORITY': 'HIGH', 'GPU_QUEUE_PRIORITY': 'HIGH'}),
    ]
    if args.nightly_benchmark_app and int8_xml is not None and int8_xml.exists():
        variants.append(('nightly_int8_sym_g128', 'Nightly benchmark_app on INT8 g128', int8_xml, args.nightly_benchmark_app, None))

    for entry in ptq_candidates:
        if entry[2].exists():
            variants.append(entry)

    baseline_avg = None
    for variant_id, recipe, model_xml, benchmark_app, props in variants:
        stats, raw = benchmark(model_xml, benchmark_app, props)
        if variant_id == 'fp16_baseline':
            baseline_avg = stats['average_ms']
        vs_fp16 = None if baseline_avg is None else ((stats['average_ms'] - baseline_avg) / baseline_avg) * 100.0
        row = {
            'variant': variant_id,
            'recipe': recipe,
            'model_xml': str(model_xml),
            'benchmark_app': str(benchmark_app),
            'average_ms': f"{stats['average_ms']:.2f}",
            'median_ms': f"{stats['median_ms']:.2f}",
            'first_inference_ms': f"{stats.get('first_inference_ms', math.nan):.2f}",
            'vs_fp16_pct': '' if vs_fp16 is None else f'{vs_fp16:.2f}',
        }
        rows.append(row)
        raw_name = f'{variant_id}.log'
        (round2_root / raw_name).write_text(raw, encoding='utf-8')

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = {}
    for variant in args.accuracy_candidates:
        candidate_xml = round2_root / variant / 'vision-encoder.xml'
        if candidate_xml.exists():
            accuracy[variant] = accuracy_check(fp16_xml, candidate_xml)
    if int8_xml is not None and int8_xml.exists():
        accuracy['int8_sym_g128'] = accuracy_check(fp16_xml, int8_xml)
    ACCURACY_JSON.write_text(json.dumps(accuracy, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
