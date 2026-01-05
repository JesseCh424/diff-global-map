#!/usr/bin/env python
"""Compare official MapTracker viz with custom viz"""

import cv2
import os
import sys

scenes = [
    '02678d04-cc9f-3148-9f95-1ba66347dff9',
    '02a00399-3857-444e-8db3-a8f58489c394',
    '04994d08-156c-3018-9717-ba0e29be8153'
]

base_dir = 'viz/av2_old/official_comparison'

for scene in scenes:
    scene_dir = os.path.join(base_dir, scene)

    # Check if both files exist
    official_path = os.path.join(scene_dir, 'pred_merged.png')
    my_path = os.path.join(scene_dir, 'my_pred_merged.png')

    if not os.path.exists(official_path):
        print(f"[SKIP] {scene}: Official viz not found")
        continue
    if not os.path.exists(my_path):
        print(f"[SKIP] {scene}: Custom viz not found")
        continue

    # Load images
    official_img = cv2.imread(official_path)
    my_img = cv2.imread(my_path)

    # Resize to same height for comparison
    h_official = official_img.shape[0]
    h_my = my_img.shape[0]

    if h_official != h_my:
        # Resize to match
        scale = h_official / h_my
        my_img = cv2.resize(my_img, (int(my_img.shape[1] * scale), h_official))

    # Concatenate side by side
    combined = cv2.hconcat([official_img, my_img])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Official MapTracker', (50, 60), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(combined, 'My Aggregation', (official_img.shape[1] + 50, 60), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # Save comparison
    comparison_path = os.path.join(scene_dir, 'comparison.png')
    cv2.imwrite(comparison_path, combined)

    print(f"[OK] {scene}: {comparison_path}")
    print(f"     Official size: {official_img.shape}")
    print(f"     My size: {my_img.shape}")
