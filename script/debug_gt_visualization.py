#!/usr/bin/env python3
"""
Debug script to investigate why GT visualization shows almost no pedestrian crossings.
"""

import os
import sys
import numpy as np
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt

# Add the poly-diffuse directory to the path
sys.path.insert(0, '/home/czhu/thesis/code/thesis_cheng/poly-diffuse')

from mmdet3d.datasets import build_dataset
from mmcv import Config
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import polygon_collate
import importlib

def debug_dataset_gt_loading():
    """Debug how GT data is loaded and processed."""
    
    # Load the config file
    config_path = '/home/czhu/thesis/code/thesis_cheng/poly-diffuse/projects/configs/maptr/maptr_tiny_r50.py'
    
    # Sample init results path (you might need to adjust this)
    init_path = '/home/czhu/thesis/code/thesis_cheng/poly-diffuse/init_results/maptr_tiny_r50_24ep_test.json'
    
    print("Loading config...")
    cfg = Config.fromfile(config_path)
    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            assert hasattr(cfg, 'plugin_dir')
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"Loading plugin: {_module_path}")
            plg_lib = importlib.import_module(_module_path)
    
    # Setup test dataset
    cfg.data.test['load_init'] = True
    cfg.data.test['init_results_path'] = init_path
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    
    pc_range = cfg.point_cloud_range
    print(f"Point cloud range: {pc_range}")
    
    # Load a few samples and analyze
    print("\nAnalyzing first few samples...")
    for i in range(min(5, len(dataset))):
        print(f"\n--- Sample {i} ---")
        try:
            sample = dataset[i]
            
            if sample is None:
                print("Sample is None, skipping")
                continue
                
            # Check GT data
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            
            print(f"GT boxes shape: {gt_bboxes_3d.shape}")
            print(f"GT labels shape: {gt_labels_3d.shape}")
            print(f"GT labels: {gt_labels_3d}")
            
            # Count by class
            divider_count = (gt_labels_3d == 0).sum().item()
            ped_crossing_count = (gt_labels_3d == 1).sum().item()
            boundary_count = (gt_labels_3d == 2).sum().item()
            
            print(f"Divider count: {divider_count}")
            print(f"Ped crossing count: {ped_crossing_count}")
            print(f"Boundary count: {boundary_count}")
            
            # Check coordinate ranges for each class
            if ped_crossing_count > 0:
                ped_indices = (gt_labels_3d == 1).nonzero(as_tuple=True)[0]
                print(f"Ped crossing indices: {ped_indices}")
                
                for idx in ped_indices:
                    ped_pts = gt_bboxes_3d[idx]
                    print(f"  Ped crossing {idx}: shape={ped_pts.shape}")
                    print(f"  X range: [{ped_pts[:, 0].min():.3f}, {ped_pts[:, 0].max():.3f}]")
                    print(f"  Y range: [{ped_pts[:, 1].min():.3f}, {ped_pts[:, 1].max():.3f}]")
                    
                    # Check if coordinates are mostly zeros or very small
                    non_zero_count = (ped_pts.abs() > 0.001).sum().item()
                    print(f"  Non-zero coordinates: {non_zero_count}/{ped_pts.numel()}")
                    
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Also check the vector map generation directly
    print("\n--- Testing vector map generation directly ---")
    sample_info = dataset.get_data_info(0)
    if sample_info:
        print(f"Sample info keys: {sample_info.keys()}")
        print(f"Location: {sample_info.get('map_location', 'unknown')}")
        
        # Test the vector map generation
        from nuscenes.eval.common.utils import Quaternion
        
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample_info['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = sample_info['ego2global_translation']
        
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        
        print(f"Lidar2global translation: {lidar2global_translation}")
        
        # Call the vector map generation
        anns_results = dataset.vector_map.gen_vectorized_samples(
            sample_info['map_location'], 
            lidar2global_translation, 
            lidar2global_rotation
        )
        
        print(f"Vector map results: {anns_results.keys()}")
        
        if 'gt_vecs_label' in anns_results:
            labels = anns_results['gt_vecs_label']
            print(f"Raw labels: {labels}")
            print(f"Label counts: divider={labels.count(0)}, ped_crossing={labels.count(1)}, boundary={labels.count(2)}")
            
        if 'gt_vecs_pts_loc' in anns_results:
            pts_loc = anns_results['gt_vecs_pts_loc']
            print(f"Points location type: {type(pts_loc)}")
            if hasattr(pts_loc, 'instance_list'):
                print(f"Number of instances: {len(pts_loc.instance_list)}")

if __name__ == "__main__":
    debug_dataset_gt_loading()