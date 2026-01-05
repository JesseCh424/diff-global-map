#!/usr/bin/env python3
"""
Verify dataset splits between MapTracker and PolyDiffuse
"""

import pickle
import json
from pathlib import Path
from typing import Set, Dict, List
import argparse

def load_pkl_file(filepath: str) -> Dict:
    """Load pickle file and return data"""
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} items")
    return data

def extract_sample_tokens_from_pkl(data: Dict) -> Set[str]:
    """Extract sample tokens from PKL annotation data"""
    tokens = set()
    
    if isinstance(data, dict):
        if 'infos' in data:
            # Standard format with 'infos' key
            for info in data['infos']:
                if 'token' in info:
                    tokens.add(info['token'])
                elif 'sample_token' in info:
                    tokens.add(info['sample_token'])
        elif 'data_infos' in data:
            # Alternative format with 'data_infos' key
            for info in data['data_infos']:
                if 'token' in info:
                    tokens.add(info['token'])
                elif 'sample_token' in info:
                    tokens.add(info['sample_token'])
        else:
            # Direct list format
            for item in data:
                if isinstance(item, dict):
                    if 'token' in item:
                        tokens.add(item['token'])
                    elif 'sample_token' in item:
                        tokens.add(item['sample_token'])
    elif isinstance(data, list):
        # Direct list format
        for item in data:
            if isinstance(item, dict):
                if 'token' in item:
                    tokens.add(item['token'])
                elif 'sample_token' in item:
                    tokens.add(item['sample_token'])
    
    return tokens

def extract_sample_tokens_from_json(filepath: str) -> Set[str]:
    """Extract sample tokens from JSON output files"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tokens = set()
    
    if 'results' in data:
        results = data['results']
        
        if isinstance(results, dict):
            # MapTracker format: {"results": {"token1": {...}, "token2": {...}}}
            tokens = set(results.keys())
        elif isinstance(results, list):
            # PolyDiffuse format: {"results": [{"sample_token": "token1"}, ...]}
            for result in results:
                if 'sample_token' in result:
                    tokens.add(result['sample_token'])
    
    print(f"  Extracted {len(tokens)} sample tokens")
    return tokens

def compare_splits(maptracker_path: str, polydiffuse_path: str):
    """Compare dataset splits between MapTracker and PolyDiffuse"""
    
    print("="*80)
    print("DATASET SPLIT COMPARISON")
    print("="*80)
    
    # Load MapTracker annotation files
    maptracker_train = load_pkl_file(f"{maptracker_path}/nuscenes_map_infos_train.pkl")
    maptracker_val = load_pkl_file(f"{maptracker_path}/nuscenes_map_infos_val.pkl")
    
    # Extract sample tokens from MapTracker
    maptracker_train_tokens = extract_sample_tokens_from_pkl(maptracker_train)
    maptracker_val_tokens = extract_sample_tokens_from_pkl(maptracker_val)
    
    print(f"\nMapTracker:")
    print(f"  Train tokens: {len(maptracker_train_tokens)}")
    print(f"  Val tokens: {len(maptracker_val_tokens)}")
    print(f"  Total tokens: {len(maptracker_train_tokens | maptracker_val_tokens)}")
    
    # Try to find PolyDiffuse annotation files
    poly_data_dir = Path(polydiffuse_path)
    possible_files = [
        "nuscenes_infos_temporal_train.pkl",
        "nuscenes_infos_temporal_val.pkl",
        "nuscenes_infos_train.pkl", 
        "nuscenes_infos_val.pkl"
    ]
    
    polydiffuse_train_tokens = set()
    polydiffuse_val_tokens = set()
    
    for filename in possible_files:
        filepath = poly_data_dir / filename
        if filepath.exists():
            print(f"\nFound PolyDiffuse file: {filepath}")
            data = load_pkl_file(str(filepath))
            tokens = extract_sample_tokens_from_pkl(data)
            
            if 'train' in filename:
                polydiffuse_train_tokens = tokens
            elif 'val' in filename:
                polydiffuse_val_tokens = tokens
    
    if polydiffuse_train_tokens or polydiffuse_val_tokens:
        print(f"\nPolyDiffuse:")
        print(f"  Train tokens: {len(polydiffuse_train_tokens)}")
        print(f"  Val tokens: {len(polydiffuse_val_tokens)}")
        print(f"  Total tokens: {len(polydiffuse_train_tokens | polydiffuse_val_tokens)}")
        
        # Compare splits
        print(f"\n" + "="*50)
        print("SPLIT COMPARISON RESULTS")
        print("="*50)
        
        # Train set comparison
        train_overlap = maptracker_train_tokens & polydiffuse_train_tokens
        train_only_maptracker = maptracker_train_tokens - polydiffuse_train_tokens
        train_only_polydiffuse = polydiffuse_train_tokens - maptracker_train_tokens
        
        print(f"\nTrain Set Comparison:")
        print(f"  Common tokens: {len(train_overlap)}")
        print(f"  Only in MapTracker: {len(train_only_maptracker)}")
        print(f"  Only in PolyDiffuse: {len(train_only_polydiffuse)}")
        print(f"  Train overlap ratio: {len(train_overlap) / max(len(maptracker_train_tokens), len(polydiffuse_train_tokens)) * 100:.1f}%")
        
        # Val set comparison  
        val_overlap = maptracker_val_tokens & polydiffuse_val_tokens
        val_only_maptracker = maptracker_val_tokens - polydiffuse_val_tokens
        val_only_polydiffuse = polydiffuse_val_tokens - maptracker_val_tokens
        
        print(f"\nVal Set Comparison:")
        print(f"  Common tokens: {len(val_overlap)}")
        print(f"  Only in MapTracker: {len(val_only_maptracker)}")
        print(f"  Only in PolyDiffuse: {len(val_only_polydiffuse)}")
        print(f"  Val overlap ratio: {len(val_overlap) / max(len(maptracker_val_tokens), len(polydiffuse_val_tokens)) * 100:.1f}%")
        
        # Cross-split contamination check
        maptracker_train_in_poly_val = maptracker_train_tokens & polydiffuse_val_tokens
        maptracker_val_in_poly_train = maptracker_val_tokens & polydiffuse_train_tokens
        
        print(f"\nCross-split Analysis:")
        print(f"  MapTracker train scenes in PolyDiffuse val: {len(maptracker_train_in_poly_val)}")
        print(f"  MapTracker val scenes in PolyDiffuse train: {len(maptracker_val_in_poly_train)}")
        
        # Overall compatibility
        total_maptracker = maptracker_train_tokens | maptracker_val_tokens
        total_polydiffuse = polydiffuse_train_tokens | polydiffuse_val_tokens
        total_overlap = total_maptracker & total_polydiffuse
        
        print(f"\nOverall Compatibility:")
        print(f"  Total scenes overlap: {len(total_overlap)} / {len(total_maptracker | total_polydiffuse)}")
        print(f"  Overall compatibility: {len(total_overlap) / len(total_maptracker | total_polydiffuse) * 100:.1f}%")
        
        # Save results for further analysis
        results = {
            'maptracker_train_tokens': list(maptracker_train_tokens),
            'maptracker_val_tokens': list(maptracker_val_tokens),
            'polydiffuse_train_tokens': list(polydiffuse_train_tokens),
            'polydiffuse_val_tokens': list(polydiffuse_val_tokens),
            'train_overlap': list(train_overlap),
            'val_overlap': list(val_overlap),
            'maptracker_train_in_poly_val': list(maptracker_train_in_poly_val),
            'maptracker_val_in_poly_train': list(maptracker_val_in_poly_train)
        }
        
        with open('/home/czhu/thesis/code/thesis_cheng/split_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: split_comparison_results.json")
    else:
        print("\nWarning: Could not find PolyDiffuse annotation files for comparison")

def analyze_output_files():
    """Analyze the output JSON files to see which scenes they contain"""
    
    print("\n" + "="*80)
    print("OUTPUT FILES ANALYSIS")
    print("="*80)
    
    # Analyze MapTracker output
    maptracker_json = "/home/czhu/thesis/code/thesis_cheng/json/maptracker_nu.json"
    if Path(maptracker_json).exists():
        maptracker_tokens = extract_sample_tokens_from_json(maptracker_json)
        print(f"\nMapTracker output contains {len(maptracker_tokens)} scenes")
        
        # Save first few tokens for reference
        sample_tokens = list(maptracker_tokens)[:10]
        print(f"Sample tokens: {sample_tokens}")
    else:
        print(f"\nMapTracker output not found at: {maptracker_json}")
    
    # Analyze PolyDiffuse MapTR output
    maptr_json = "/home/czhu/thesis/code/thesis_cheng/json/maptr_test.json"
    if Path(maptr_json).exists():
        try:
            maptr_tokens = extract_sample_tokens_from_json(maptr_json)
            print(f"\nMapTR output contains {len(maptr_tokens)} scenes")
        except Exception as e:
            print(f"\nError reading MapTR output: {e}")
    else:
        print(f"\nMapTR output not found at: {maptr_json}")

def main():
    parser = argparse.ArgumentParser(description='Verify dataset splits between MapTracker and PolyDiffuse')
    parser.add_argument('--maptracker-path', default='/home/czhu/thesis/code/thesis_cheng/maptracker/datasets/nuscenes',
                       help='Path to MapTracker dataset annotation files')
    parser.add_argument('--polydiffuse-path', default='/home/czhu/thesis/code/thesis_cheng/poly-diffuse/data/nuscenes',
                       help='Path to PolyDiffuse dataset annotation files')
    
    args = parser.parse_args()
    
    try:
        compare_splits(args.maptracker_path, args.polydiffuse_path)
        analyze_output_files()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()