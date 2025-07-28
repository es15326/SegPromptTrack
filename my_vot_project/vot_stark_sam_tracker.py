#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Unified STARK+SAM tracker for VOT challenges.

This script combines a STARK-based bounding box tracker with the Segment Anything
Model (SAM) for mask generation. It is designed to be highly configurable via
command-line arguments to facilitate different experimental setups.
"""

import argparse
import gc
import os
import sys

import cv2
import numpy as np
import torch
import vot

# --- Setup Project Path ---
# Assumes the script is run from a location where this relative path is valid,
# or the user provides an absolute path.
STARK_PROJECT_PATH = '/usr/mvl2/esdft/Stark/'
if STARK_PROJECT_PATH not in sys.path:
    sys.path.append(STARK_PROJECT_PATH)

from lib.test.evaluation import Tracker as StarkTrackerFactory
from segment_anything import sam_model_registry, SamPredictor
from vot_data_preprocessing import _mask_to_bbox, get_bbox

class ConfigurableTracker:
    """
    A configurable tracker that combines a bounding box tracker (STARK)
    with a segmentation model (SAM).
    """

    def __init__(self, tracker_name, tracker_param, vot_version):
        """Initializes the STARK tracker with specified parameters."""
        print(f"Initializing STARK tracker: {tracker_name} ({tracker_param}) for {vot_version}")
        tracker_info = StarkTrackerFactory(tracker_name, tracker_param, vot_version, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, image_rgb, initial_bbox):
        """Initializes the tracker with the first frame and a bounding box."""
        self.tracker.initialize(image_rgb, {'init_bbox': initial_bbox})

    def track_box(self, image_rgb):
        """Tracks the object in the given image to get a bounding box."""
        return self.tracker.track(image_rgb)['target_bbox']

def main():
    """
    Main function to parse arguments and run the multi-object tracking loop.
    """
    parser = argparse.ArgumentParser(description="Run a configurable STARK+SAM tracker for VOT.")
    
    # --- STARK Tracker Arguments ---
    parser.add_argument('--tracker_name', type=str, required=True, help="Name of the STARK tracker (e.g., 'stark_s', 'stark_st').")
    parser.add_argument('--tracker_param', type=str, default='baseline', help="Parameter set for the tracker (e.g., 'baseline', 'baseline_R101').")
    
    # --- SAM Model Arguments ---
    parser.add_argument('--sam_checkpoint', type=str, required=True, help="Path to the SAM model checkpoint.")
    parser.add_argument('--sam_model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], help="SAM model architecture.")
    
    # --- VOT Toolkit Arguments ---
    parser.add_argument('--vot_version', type=str, default='vot20', help="VOT challenge version (e.g., 'vot20', 'vot20lt').")
    
    # --- Masking Strategy Arguments ---
    parser.add_argument('--multimask_output', action='store_true', help="If set, use SAM's multimask output and select the best one.")
    
    # --- System Arguments ---
    parser.add_argument('--gpu_id', type=str, default='3', help="ID of the GPU to use.")

    args = parser.parse_args()

    # --- Setup Environment ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = "cuda"
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found at: {args.sam_checkpoint}")

    # --- Initialize Models ---
    print(f"Loading SAM model: {args.sam_model_type} from {os.path.basename(args.sam_checkpoint)}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # --- VOT Handle ---
    handle = vot.VOT("mask", multiobject=True)
    
    try:
        initial_frame_path = handle.frame()
        if not initial_frame_path:
            return

        initial_image_rgb = cv2.cvtColor(cv2.imread(initial_frame_path), cv2.COLOR_BGR2RGB)
        initial_masks = handle.objects()

        # Initialize one tracker instance for each object.
        trackers = [ConfigurableTracker(args.tracker_name, args.tracker_param, args.vot_version) for _ in initial_masks]
        for tracker, mask in zip(trackers, initial_masks):
            tracker.initialize(initial_image_rgb, _mask_to_bbox(mask))

        # --- Main Tracking Loop ---
        while True:
            current_frame_path = handle.frame()
            if not current_frame_path:
                break

            current_image_rgb = cv2.cvtColor(cv2.imread(current_frame_path), cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(current_image_rgb)

            predicted_masks = []
            for tracker in trackers:
                bbox = tracker.track_box(current_image_rgb)
                input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                # Generate mask using SAM based on the chosen strategy.
                masks, scores, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=args.multimask_output,
                )
                
                if args.multimask_output:
                    # Select the mask with the highest score.
                    best_mask = masks[np.argmax(scores)]
                    final_mask = best_mask.astype(np.uint8)
                else:
                    # Use the single mask output.
                    final_mask = (masks[0] * 1).astype(np.uint8)

                predicted_masks.append(final_mask)

            handle.report(predicted_masks)
            
            # Optional: Clear cache to manage memory
            torch.cuda.empty_cache()
            gc.collect()

    finally:
        handle.quit()

if __name__ == "__main__":
    main()
