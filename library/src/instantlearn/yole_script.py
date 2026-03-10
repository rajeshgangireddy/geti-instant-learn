from instantlearn.models import YOLOE, YOLOEOpenVINO
from instantlearn.data import Sample
from pathlib import Path
import time
import argparse

example_path = Path("examples/assets/coco")


default_variant =  "yoloe-26x-seg"


def infer_n_times(model, samples, n=20):
    # cycle though samples for 20 times
    infer_times = []
    for i in range(n):
        sample = samples[i % len(samples)]
        start_time = time.time()
        model.predict(sample)
        elapsed = time.time() - start_time
        infer_times.append(elapsed)

    return infer_times

def main(args):
    device = args.device
    torch_model = YOLOE(model_name=default_variant, device=device)
    reference_sample = Sample(
    image_path="examples/assets/coco/000000286874.jpg",
    mask_paths="examples/assets/coco/000000286874_mask.png")

    tic = time.time()
    torch_model.fit(reference_sample)
    toc = time.time()
    print(f"Fitting time: {toc - tic:.3f} seconds.")
    
    print(f"Inference on target images with PyTorch model...")
    target_images = ["000000173279.jpg", "000000390341.jpg","000000286874.jpg","000000267704.jpg"]
    target_samples = [Sample(image_path=example_path / target) for target in target_images]

    torch_model.predict(target_samples[0])  # warmup

    torch_times = infer_n_times(torch_model, target_samples, n=args.num_iters)

    

    # Export to OpenVINO
    print(f"\nExporting to OpenVINO IR...")
    # use a tmp directory for export outputs
    export_dir = Path("exports/yoloe_openvino_demo")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = torch_model.export(export_dir=export_dir, backend="openvino")
    print(f"Exported OpenVINO IR to: {export_path}")

    # Load OpenVINO model
    print("\nLoading OpenVINO model...")
    ov_model = YOLOEOpenVINO(model_dir=export_path, device="cpu")
    ov_model.fit(reference_sample)

    print("Inference on target images with OpenVINO model...")
    target_samples = [Sample(image_path=example_path / target) for target in target_images]

    # Warmup OpenVINO (first inferences include compilation overhead)
    ov_model.predict(target_samples[0])

    print("Inference on target images with OpenVINO model...")
    ov_times = infer_n_times(ov_model, target_samples, n=args.num_iters)

    #remove  fastest and slowest times to reduce outlier effect
    torch_times = sorted(torch_times)[1:-1]
    print(f"Average Torch {len(torch_times)} runs: {sum(torch_times) / len(torch_times):.3f} seconds.")

    # same for openvino times
    ov_times = sorted(ov_times)[1:-1]
    print(f"Average OpenVINO {len(ov_times)} runs: {sum(ov_times) / len(ov_times):.3f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOE export example")
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--num_iters", type=int, default=20)
    
    args = parser.parse_args()
    main(args) 