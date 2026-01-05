#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert maptracker_nu.json format to maptr_test.json format or generate.py format
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

def convert_maptracker_to_maptr(input_file: str, output_file: str):
    """
    Convert maptracker format to maptr format
    
    maptracker format:
    {
        "meta": {...},
        "results": {
            "token1": {"vectors": [...], ...},
            "token2": {"vectors": [...], ...},
            ...
        }
    }
    
    maptr format:
    {
        "meta": {...},
        "results": [
            {"sample_token": "token1", "vectors": [...], ...},
            {"sample_token": "token2", "vectors": [...], ...},
            ...
        ]
    }
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        maptracker_data = json.load(f)
    
    print("Converting format...")
    
    # Create new maptr format structure
    maptr_data = {
        "meta": maptracker_data["meta"].copy(),
        "results": []
    }
    
    # Convert results from dict to list
    results_dict = maptracker_data["results"]
    
    for sample_token, sample_data in results_dict.items():
        # Create new result entry with sample_token as a field
        new_entry = {
            "sample_token": sample_token
        }
        
        # Copy all other fields from the original data
        for key, value in sample_data.items():
            new_entry[key] = value
        
        maptr_data["results"].append(new_entry)
    
    print(f"Converted {len(maptr_data['results'])} entries")
    
    # Save converted data
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(maptr_data, f, ensure_ascii=False, separators=(',', ':'))
    
    print("Conversion completed successfully!")


def analyze_maptracker_labels(input_file: str):
    """
    Analyze maptracker labels to understand their encoding
    """
    print(f"Analyzing label encoding in {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        maptracker_data = json.load(f)
    
    results_dict = maptracker_data["results"]
    
    # Analyze several samples to understand label semantics
    sample_count = 0
    label_examples = {0: [], 1: [], 2: []}
    
    for sample_token, sample_data in results_dict.items():
        if sample_count >= 5:  # Analyze first 5 samples
            break
            
        vectors = np.array(sample_data['vectors'])  # Shape: (N, 20, 2)
        labels = np.array(sample_data['labels'])    # Shape: (N,)
        
        for label_type in [0, 1, 2]:
            indices = np.where(labels == label_type)[0]
            if len(indices) > 0:
                for idx in indices[:2]:  # Take first 2 examples per label per sample
                    example_vector = vectors[idx]
                    label_examples[label_type].append({
                        'sample': sample_token,
                        'vector_idx': idx,
                        'x_range': [example_vector[:, 0].min(), example_vector[:, 0].max()],
                        'y_range': [example_vector[:, 1].min(), example_vector[:, 1].max()],
                        'x_span': example_vector[:, 0].max() - example_vector[:, 0].min(),
                        'y_span': example_vector[:, 1].max() - example_vector[:, 1].min()
                    })
        
        sample_count += 1
    
    # Analyze patterns
    print("\nLabel analysis:")
    for label_type in [0, 1, 2]:
        examples = label_examples[label_type]
        if examples:
            x_spans = [ex['x_span'] for ex in examples]
            y_spans = [ex['y_span'] for ex in examples]
            
            print(f"\nLabel {label_type} ({len(examples)} examples):")
            print(f"  Average X span: {np.mean(x_spans):.2f} ± {np.std(x_spans):.2f}")
            print(f"  Average Y span: {np.mean(y_spans):.2f} ± {np.std(y_spans):.2f}")
            print(f"  X span range: [{min(x_spans):.2f}, {max(x_spans):.2f}]")
            print(f"  Y span range: [{min(y_spans):.2f}, {max(y_spans):.2f}]")
            
            # Show first few examples
            for i, ex in enumerate(examples[:3]):
                print(f"    Example {i+1}: X[{ex['x_range'][0]:.1f}, {ex['x_range'][1]:.1f}], Y[{ex['y_range'][0]:.1f}, {ex['y_range'][1]:.1f}]")
    
    # Suggest label mapping based on analysis
    print("\n" + "="*50)
    print("LABEL MAPPING ANALYSIS:")
    print("="*50)
    
    # Based on typical lane geometry:
    # - dividers are usually long longitudinal lines (large Y span, small X span)
    # - ped_crossings are usually shorter transverse lines (large X span, smaller Y span)  
    # - boundaries are usually long lines along road edges
    
    return label_examples


def convert_maptracker_to_generate_format(input_file: str, output_file: str, analyze_labels: bool = False):
    """
    Convert maptracker_nu.json directly to generate.py compatible format
    
    This function:
    1. Converts from dict format to list format
    2. Applies coordinate transformation to match generate.py expectations
    3. Structures data to match maptr_test.json format
    4. Optionally analyzes and fixes label mapping
    
    IMPORTANT: generate.py expects real world coordinates (not normalized),
    because preprocess_init_result() will normalize them by dividing by pc_range.
    
    Args:
        input_file: Path to maptracker_nu.json
        output_file: Path to save generate.py compatible JSON file
        analyze_labels: Whether to analyze label patterns first
    """
    if analyze_labels:
        analyze_maptracker_labels(input_file)
        
        response = input("\nDo you want to continue with conversion? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        maptracker_data = json.load(f)
    
    print("Converting to generate.py compatible format...")
    
    # Extract results from the original dict format
    results_dict = maptracker_data["results"]
    print(f"Converting {len(results_dict)} samples...")
    
    converted_results = []
    
    # LABEL MAPPING - Fixed based on analysis
    # MapTracker encoding: 0=ped_crossing, 1=divider, 2=boundary  
    # PolyDiffuse expects: 0=divider, 1=ped_crossing, 2=boundary
    # So we need to remap: maptracker_0->polydiffuse_1, maptracker_1->polydiffuse_0, maptracker_2->polydiffuse_2
    
    maptracker_to_polydiffuse = {
        0: 1,  # maptracker ped_crossing -> polydiffuse ped_crossing (type 1)
        1: 0,  # maptracker divider -> polydiffuse divider (type 0) 
        2: 2,  # maptracker boundary -> polydiffuse boundary (type 2)
    }
    
    label_names = ['divider', 'ped_crossing', 'boundary']  # polydiffuse order
    
    for sample_token, sample_data in tqdm(results_dict.items(), desc="Converting samples"):
        # Extract vectors, labels, and scores from the sample data
        vectors = np.array(sample_data['vectors'])  # Shape: (N, 20, 2)
        labels = np.array(sample_data['labels'])    # Shape: (N,)
        scores = np.array(sample_data['scores'])    # Shape: (N,)
        
        # Convert to maptr_test.json format with coordinate transformation
        converted_vectors = []
        
        for i, (vector, label, score) in enumerate(zip(vectors, labels, scores)):
            # CORRECTED: Keep coordinates in real world scale for generate.py
            # Current maptracker data: X∈[-30, 30], Y∈[-15, 15]
            # Expected for generate.py: real world coordinates that will be normalized by preprocess_init_result
            
            pts_corrected = vector.copy()
            
            # Apply coordinate conversion to match expected coordinate system
            # maptracker: X=left-right, Y=front-back
            # generate.py/MapTR: X=front-back, Y=left-right  
            pts_corrected[:, 0] = -vector[:, 1]  # new X = -old Y 
            pts_corrected[:, 1] = vector[:, 0]   # new Y = old X
            
            # DO NOT normalize - keep in real world coordinates
            # generate.py's preprocess_init_result() will do the normalization:
            # pts[:, 0] /= pc_range[3]  (divide by 15.0)
            # pts[:, 1] /= pc_range[4]  (divide by 30.0)
            
            # Apply label remapping
            polydiffuse_label = maptracker_to_polydiffuse[label]
            
            # Create vector entry in maptr_test.json format
            vector_entry = {
                'pts': pts_corrected.tolist(),
                'pts_num': 20,
                'cls_name': label_names[polydiffuse_label],
                'type': int(polydiffuse_label),
                'confidence_level': float(score)
            }
            
            converted_vectors.append(vector_entry)
        
        converted_sample = {
            'sample_token': sample_token,
            'vectors': converted_vectors
        }
        
        converted_results.append(converted_sample)
    
    # Create final structure matching maptr_test.json
    converted_data = {
        'meta': maptracker_data['meta'] if 'meta' in maptracker_data else {},
        'results': converted_results
    }
    
    print(f"Saving converted data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, separators=(',', ':'))
    
    print(f"Conversion completed!")
    print(f"Original samples: {len(results_dict)}")
    print(f"Converted samples: {len(converted_results)}")
    
    # Verify the conversion by checking coordinate ranges
    if len(converted_results) > 0:
        sample_vectors = converted_results[0]['vectors']
        if len(sample_vectors) > 0:
            pts = np.array([v['pts'] for v in sample_vectors])
            print(f"Converted coordinate ranges:")
            print(f"  X: [{pts[:,:,0].min():.3f}, {pts[:,:,0].max():.3f}]")
            print(f"  Y: [{pts[:,:,1].min():.3f}, {pts[:,:,1].max():.3f}]")
            print(f"Expected: similar to maptr_test.json (real world coordinates)")
            print(f"generate.py will normalize these using preprocess_init_result()")
            print("\nData is now ready for use with generate.py!")


def main():
    parser = argparse.ArgumentParser(description='Convert maptracker format to maptr format or generate.py format')
    parser.add_argument('--input', '-i', 
                       default='/home/czhu/thesis/code/thesis_cheng/json/maptracker_nu.json',
                       help='Input maptracker JSON file')
    parser.add_argument('--output', '-o',
                       default='/home/czhu/thesis/code/thesis_cheng/json/maptracker_nu_converted.json', 
                       help='Output JSON file')
    parser.add_argument('--mode', '-m',
                       choices=['maptr', 'generate'], 
                       default='maptr',
                       help='Conversion mode: "maptr" for basic format conversion, "generate" for generate.py compatible format with coordinate transformation')
    
    args = parser.parse_args()
    
    print("MapTracker Format Converter")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Mode: {args.mode}")
    print()
    
    try:
        if args.mode == 'generate':
            convert_maptracker_to_generate_format(args.input, args.output)
        else:  # maptr mode (default)
            convert_maptracker_to_maptr(args.input, args.output)
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

if __name__ == "__main__":
    main() 