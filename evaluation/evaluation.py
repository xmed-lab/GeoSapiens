import json
import os
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment as assign
from tqdm import tqdm
import sys
import argparse


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
default_annotations = os.path.join(root_dir, 'data/teeth/annotations/annotations_test.json')

from utils.landmark_statistics import LandmarkStatistics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument(
        '--predictions',
        help='Path to the predictions dump.pkl file',
        # You should change this to the path to the predictions dump.pkl file
        default=''
    )
    parser.add_argument(
        '--annotations',
        default=default_annotations,
        help='Path to the ground truth annotations file'
    )
    parser.add_argument(
        '--output',
        help='Path to save the evaluation results'
    )
    return parser.parse_args()

# Evaluation function
def evaluate(annotations_path, predictions_path, keypoint_order=None):
    with open(annotations_path, 'r') as f:
        gt_data = json.load(f)

    with open(predictions_path, 'rb') as f:
        pred_data = pickle.load(f)

    # Map ground truth annotations by image ID
    gt_annotations = {ann['image_id']: np.array(ann['keypoints']).reshape(-1, 3)[:, :2] for ann in
                      gt_data['annotations']}
    # Map predictions by image filename
    pred_annotations = {entry['img_path'].split('/')[-1]: np.array(entry['pred_instances']['keypoints']) for entry
                        in
                        pred_data}
    landmark_statistic = LandmarkStatistics()
    for gt_image in tqdm(gt_data['images'], desc="Evaluating"):
        image_id = gt_image['id']
        file_name = gt_image['file_name']

        gt_keypoints = gt_annotations[image_id]
        if keypoint_order:
            pred_keypoints = pred_annotations[file_name][0][keypoint_order]
        else:
            pred_keypoints = pred_annotations[file_name][0]

        # teeth
        radii = [0.5, 1.0, 2.0]
        
        spacing = [1, 1]
        spacing = [1/5.5556, 1/5.5556]
        
        
        landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints, spacing=spacing)
        

    overview_string = landmark_statistic.get_overview_string(radii)
    return overview_string




def parse_results(raw_output):
    """
    Parse the raw output string to extract meaningful metrics, including percentages of outliers.

    Args:
        raw_output (str): The raw evaluation output string.

    Returns:
        dict: A dictionary of parsed metrics.
    """
    # print(raw_output)
    results = {}
    lines = raw_output.splitlines()

    # Initialize total number of GT points
    total_gt_points = 0

    # Extract total number of GT points
    for line in lines:
        if "valid gt" in line:
            total_gt_points = int(line.split("out of")[1].split("valid gt")[0].strip())
            break

    # Parse PE values and percentages
    for line in lines:
        if line.startswith("mean:"):
            results["MRE"] = float(line.split(":")[1].strip())
        elif line.startswith("std:"):
            results["std"] = float(line.split(":")[1].strip())
        # elif line.startswith("median:"):
        #     results["median"] = float(line.split(":")[1].strip())
        elif line.startswith("#outliers >= 0.3:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_0.3"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 0.5:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_0.5"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 1.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_1.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 2.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_2.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 3.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_3.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 3.1:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_3.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 4.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_4.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 5.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_5.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 6.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_6.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 9.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_9.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 10.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_10.0"] = 100 - (outlier_count / total_gt_points) * 100

    return results


# Main function
if __name__ == "__main__":
    args = parse_args()
    
    raw_output = evaluate(args.annotations, args.predictions)
    parsed_results = parse_results(raw_output)
    
    print("\nEvaluation Results:")
    for key, value in parsed_results.items():
        print(f"{key}: {value}")
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Evaluation Results:\n")
            for key, value in parsed_results.items():
                f.write(f"{key}: {value}\n")
