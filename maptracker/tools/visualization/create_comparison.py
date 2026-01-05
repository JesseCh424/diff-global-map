#!/usr/bin/env python
"""Create side-by-side comparison images"""

import cv2
import os
import sys

scenes = [
    '02678d04-cc9f-3148-9f95-1ba66347dff9',
    '02a00399-3857-444e-8db3-a8f58489c394',
    '04994d08-156c-3018-9717-ba0e29be8153'
]

base_dir = 'viz/av2_old'

for scene in scenes:
    scene_dir = os.path.join(base_dir, scene)

    official_path = os.path.join(scene_dir, 'official_pred_merged.png')
    my_path = os.path.join(scene_dir, 'my_pred_merged.png')

    if not os.path.exists(official_path):
        print(f"[SKIP] {scene}: Official not found")
        continue
    if not os.path.exists(my_path):
        print(f"[SKIP] {scene}: Mine not found")
        continue

    # Load images
    official_img = cv2.imread(official_path)
    my_img = cv2.imread(my_path)

    print(f"\n{scene}:")
    print(f"  Official: {official_img.shape}")
    print(f"  Mine:     {my_img.shape}")

    # Resize to same height for comparison
    h_max = max(official_img.shape[0], my_img.shape[0])

    if official_img.shape[0] != h_max:
        scale = h_max / official_img.shape[0]
        official_img = cv2.resize(official_img, (int(official_img.shape[1] * scale), h_max))

    if my_img.shape[0] != h_max:
        scale = h_max / my_img.shape[0]
        my_img = cv2.resize(my_img, (int(my_img.shape[1] * scale), h_max))

    # Concatenate side by side
    combined = cv2.hconcat([official_img, my_img])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Official (pos_predictions_5.pkl)', (50, 60), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(combined, 'Mine (aggregated .pkl)', (official_img.shape[1] + 50, 60), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Save comparison
    comparison_path = os.path.join(scene_dir, 'comparison.png')
    cv2.imwrite(comparison_path, combined)

    print(f"  Comparison: {comparison_path}")
