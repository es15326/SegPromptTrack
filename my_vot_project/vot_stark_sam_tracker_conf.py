#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This script implements a multi-object tracker using the STARK tracker for bounding box
prediction and the Segment Anything Model (SAM) for mask generation. It is designed
to integrate with the VOT toolkit for evaluation.

The confidence threshold for SAM's mask prediction can be set via a command-line argument.
"""

import os
import sys
import cv2
import numpy as np
import vot
import argparse # Import the argparse library

# --- Project-specific imports ---
STARK_PROJECT_PATH = '/usr/mvl2/esdft/Stark/'
if STARK_PROJECT_PATH not in sys.path:
    sys.path.append(STARK_PROJECT_PATH)

from lib.test.evaluation import Tracker as StarkTrackerFactory
from segment_anything import sam_model_registry, SamPredictor
from vot_data_preprocessing import _mask_to_bbox

# --- Constants and Configuration ---
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

SAM_CHECKPOINT_PATH = "/usr/mvl2/esdft/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cuda"

class MultiObjectTracker:
    """
    A tracker that combines a bounding box tracker (STARK) with a segmentation model (SAM).
    """

    def __init__(self, tracker_name='stark_st', tracker_param='baseline', visualization=False, debug=False):
        """Initializes the STARK tracker."""
        tracker_info = StarkTrackerFactory(tracker_name, tracker_param, "vot20", None)
        params = tracker_info.get_parameters()
        params.visualization = visualization
        params.debug = debug
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, image_rgb, initial_bbox):
        """Initializes the tracker with the first frame and a bounding box."""
        self.tracker.initialize(image_rgb, {'init_bbox': initial_bbox})

    def track_bounding_box(self, image_rgb):
        """Tracks the object in the given image to get a bounding box."""
        prediction = self.tracker.track(image_rgb)
        return prediction['target_bbox']


def initialize_sam_predictor(model_path, model_type, device):
    """Loads the SAM model and initializes the predictor."""
    print("Initializing Segment Anything Model (SAM)...")
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM initialization complete.")
    return predictor


def main():
    """
    Main function to run the multi-object tracking loop using the VOT toolkit.
    """
    # --- Argument Parsing ---
    # Set up the parser to read command-line arguments
    parser = argparse.ArgumentParser(description="Run STARK+SAM tracker with a configurable confidence threshold.")
    parser.add_argument(
        '-c', '--confidence', 
        type=float, 
        default=0.6, 
        help='Confidence threshold for accepting a mask from SAM (default: 0.6).'
    )
    args = parser.parse_args()
    print(f"Using confidence threshold: {args.confidence}")

    handle = vot.VOT("mask", multiobject=True)
    
    try:
        # --- Initialization ---
        initial_frame_path = handle.frame()
        if not initial_frame_path:
            return

        initial_image_rgb = cv2.cvtColor(cv2.imread(initial_frame_path), cv2.COLOR_BGR2RGB)
        initial_masks = handle.objects()

        trackers = [MultiObjectTracker() for _ in initial_masks]
        for tracker, mask in zip(trackers, initial_masks):
            initial_bbox = _mask_to_bbox(mask)
            tracker.initialize(initial_image_rgb, initial_bbox)

        sam_predictor = initialize_sam_predictor(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, DEVICE)

        # --- Main Tracking Loop ---
        while True:
            current_frame_path = handle.frame()
            if not current_frame_path:
                break

            current_image_rgb = cv2.cvtColor(cv2.imread(current_frame_path), cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(current_image_rgb)

            predicted_masks = []
            for tracker in trackers:
                bbox = tracker.track_bounding_box(current_image_rgb)
                input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                masks, iou_predictions, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Use the confidence threshold from the command-line argument
                if iou_predictions[0] > args.confidence:
                    final_mask = (masks[0] * 1).astype(np.uint8)
                else:
                    final_mask = np.zeros_like(masks[0], dtype=np.uint8)
                
                predicted_masks.append(final_mask)

            handle.report(predicted_masks)

    finally:
        handle.quit()


if __name__ == "__main__":
    main()
