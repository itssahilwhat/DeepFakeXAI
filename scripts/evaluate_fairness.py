import argparse
import json
from collections import defaultdict

def evaluate_fairness(input_json, output_json):
    data = json.load(open(input_json))
    group_counts = defaultdict(lambda: {"total": 0, "fake": 0})
    for rec in data:
        eth = rec.get("ethnicity", "unknown")
        is_fake = rec.get("is_fake", False)
        group_counts[eth]["total"] += 1
        if is_fake:
            group_counts[eth]["fake"] += 1
    fairness = {}
    for eth, vals in group_counts.items():
        rate = vals["fake"] / vals["total"] if vals["total"] > 0 else 0
        fairness[eth] = {"fake_rate": rate, "count": vals["total"]}
    with open(output_json, "w") as f:
        json.dump(fairness, f, indent=2)
    print(f"Fairness breakdown saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fairness by ethnicity")
    parser.add_argument("-i", "--input", required=True, help="Path to results JSON")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    evaluate_fairness(args.input, args.output)
