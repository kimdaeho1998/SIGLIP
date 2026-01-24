import argparse
import json
import sys

try:
    import torch
    import torch.nn.functional as F
    from transformers import SiglipModel, SiglipProcessor
except Exception as exc:
    print(
        "Missing dependencies. Install torch and transformers to run this script.",
        file=sys.stderr,
    )
    raise


def compute_similarity(current_summary, previous_summary, model_name):
    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    model.eval()

    inputs = processor(
        text=[current_summary, previous_summary],
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=-1)
    similarity = F.cosine_similarity(text_features[0], text_features[1], dim=0).item()
    return similarity


def build_review(similarity, threshold):
    status = "PASS" if similarity >= threshold else "FAIL"
    if status == "PASS":
        decision = "Current summary is consistent with previous summary."
    else:
        decision = "Current summary diverges from previous summary."
    return {
        "cosine_similarity": round(similarity, 4),
        "threshold": threshold,
        "status": status,
        "review": decision,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIGLIP text consistency check using cosine similarity."
    )
    parser.add_argument("--current", required=True, help="Current stage summary")
    parser.add_argument("--previous", required=True, help="Previous stage summary")
    parser.add_argument(
        "--model",
        default="google/siglip-base-patch16-224",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Pass threshold for cosine similarity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    similarity = compute_similarity(args.current, args.previous, args.model)
    result = build_review(similarity, args.threshold)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
