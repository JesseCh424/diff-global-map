#!/usr/bin/env python3
"""
Visualize coordinate comparison: Original MapTracker, Converted MapTracker, and MapTR
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random

# Colors for visualization
COLORS = {
    'divider': 'orange',
    'ped_crossing': 'blue', 
    'boundary': 'red'
}

# Label mappings
MAPTRACKER_LABELS = {0: 'ped_crossing', 1: 'divider', 2: 'boundary'}
MAPTR_LABELS = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}

def load_data():
    """Load both JSON files"""
    with open('/home/czhu/thesis/code/thesis_cheng/json/maptracker_nu.json', 'r') as f:
        mt_data = json.load(f)
    with open('/home/czhu/thesis/code/thesis_cheng/json/maptr_test.json', 'r') as f:
        mr_data = json.load(f)
    return mt_data, mr_data

def apply_coordinate_conversion(vectors):
    """Apply coordinate conversion: X=-oldY, Y=oldX"""
    converted = []
    for vec in vectors:
        new_vec = vec.copy()
        new_vec[:, 0] = -vec[:, 1]  # X = -old Y 
        new_vec[:, 1] = vec[:, 0]   # Y = old X
        converted.append(new_vec)
    return np.array(converted)

def apply_label_conversion(labels):
    """Convert MapTracker labels to MapTR format"""
    # MapTracker: 0=ped_crossing, 1=divider, 2=boundary
    # MapTR: 0=divider, 1=ped_crossing, 2=boundary
    conversion_map = {0: 1, 1: 0, 2: 2}
    return [conversion_map[label] for label in labels]

def extract_sample_data(mt_data, mr_data, sample_token, confidence_threshold=0.3):
    """Extract and process data for a specific sample"""
    
    # MapTracker data
    mt_sample = mt_data['results'][sample_token]
    mt_vectors = np.array(mt_sample['vectors'])
    mt_labels = np.array(mt_sample['labels'])
    mt_scores = np.array(mt_sample['scores'])
    
    # Filter by confidence
    mt_valid = mt_scores > confidence_threshold
    mt_vectors = mt_vectors[mt_valid]
    mt_labels = mt_labels[mt_valid]
    mt_scores = mt_scores[mt_valid]
    
    # Apply coordinate conversion
    mt_converted = apply_coordinate_conversion(mt_vectors)
    
    # Apply label conversion
    mt_labels_converted = apply_label_conversion(mt_labels)
    
    # Get label names
    mt_label_names = [MAPTRACKER_LABELS[label] for label in mt_labels]
    mt_label_names_converted = [MAPTR_LABELS[label] for label in mt_labels_converted]
    
    # MapTR data
    mr_sample = None
    for sample in mr_data['results']:
        if sample['sample_token'] == sample_token:
            mr_sample = sample
            break
    
    if not mr_sample:
        return None, None, None, None, None, None, None, None, None
    
    mr_vectors = []
    mr_labels = []
    mr_scores = []
    mr_label_names = []
    
    for v in mr_sample['vectors']:
        if v['confidence_level'] > confidence_threshold:
            mr_vectors.append(np.array(v['pts']))
            mr_labels.append(v['type'])
            mr_scores.append(v['confidence_level'])
            mr_label_names.append(v['cls_name'])
    
    mr_vectors = np.array(mr_vectors) if mr_vectors else np.array([])
    mr_labels = np.array(mr_labels) if mr_labels else np.array([])
    mr_scores = np.array(mr_scores) if mr_scores else np.array([])
    
    return (mt_vectors, mt_converted, mr_vectors, 
            mt_label_names, mt_label_names_converted, mr_label_names,
            mt_scores, mr_scores, mt_labels_converted)

def create_coordinate_comparison_plot(sample_token, mt_vectors, mt_converted, mr_vectors,
                                    mt_label_names, mt_label_names_converted, mr_label_names,
                                    mt_scores, mr_scores, output_dir):
    """Create three-panel comparison plot"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Coordinate Comparison - {sample_token[:12]}...', fontsize=16)
    
    # Panel 1: Original MapTracker
    ax1.set_title('MapTracker Original\n(X=left-right, Y=front-back)')
    
    for i, (vec, label, score) in enumerate(zip(mt_vectors, mt_label_names, mt_scores)):
        if len(vec) > 0:
            color = COLORS.get(label, 'gray')
            ax1.plot(vec[:, 0], vec[:, 1], color=color, linewidth=2, alpha=0.8)
            # Add confidence annotation for first few vectors
            if i < 3:
                ax1.text(vec[0, 0], vec[0, 1], f'{score:.2f}', 
                        fontsize=8, color=color, weight='bold')
    
    ax1.set_xlim(-35, 35)
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel('X (left-right, meters)')
    ax1.set_ylabel('Y (front-back, meters)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add legend
    handles = [plt.Line2D([0], [0], color=COLORS[label], linewidth=2, label=label) 
               for label in COLORS.keys()]
    ax1.legend(handles=handles, loc='upper right')
    
    # Add info box
    ax1.text(0.02, 0.98, f'Vectors: {len(mt_vectors)}', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Converted MapTracker
    ax2.set_title('MapTracker Converted\n(X=front-back, Y=left-right)')
    
    for i, (vec, label, score) in enumerate(zip(mt_converted, mt_label_names_converted, mt_scores)):
        if len(vec) > 0:
            color = COLORS.get(label, 'gray')
            ax2.plot(vec[:, 0], vec[:, 1], color=color, linewidth=2, alpha=0.8)
            # Add confidence annotation for first few vectors
            if i < 3:
                ax2.text(vec[0, 0], vec[0, 1], f'{score:.2f}', 
                        fontsize=8, color=color, weight='bold')
    
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-35, 35)
    ax2.set_xlabel('X (front-back, meters)')
    ax2.set_ylabel('Y (left-right, meters)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add legend
    ax2.legend(handles=handles, loc='upper right')
    
    # Add info box
    ax2.text(0.02, 0.98, f'Vectors: {len(mt_converted)}', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 3: MapTR Original
    ax3.set_title('MapTR Original\n(X=front-back, Y=left-right)')
    
    for i, (vec, label, score) in enumerate(zip(mr_vectors, mr_label_names, mr_scores)):
        if len(vec) > 0:
            color = COLORS.get(label, 'gray')
            ax3.plot(vec[:, 0], vec[:, 1], color=color, linewidth=2, alpha=0.8)
            # Add confidence annotation for first few vectors
            if i < 3:
                ax3.text(vec[0, 0], vec[0, 1], f'{score:.2f}', 
                        fontsize=8, color=color, weight='bold')
    
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-35, 35)
    ax3.set_xlabel('X (front-back, meters)')
    ax3.set_ylabel('Y (left-right, meters)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Add legend
    ax3.legend(handles=handles, loc='upper right')
    
    # Add info box
    ax3.text(0.02, 0.98, f'Vectors: {len(mr_vectors)}', transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / f'coordinate_comparison_{sample_token}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved coordinate comparison: {output_path}")
    
    return output_path

def analyze_coordinate_ranges(mt_vectors, mt_converted, mr_vectors, sample_token):
    """Analyze and print coordinate ranges"""
    
    print(f"\n=== Coordinate Analysis for {sample_token[:12]}... ===")
    
    if len(mt_vectors) > 0:
        mt_all = np.concatenate([vec for vec in mt_vectors], axis=0)
        print(f"MapTracker Original: X[{mt_all[:, 0].min():.1f}, {mt_all[:, 0].max():.1f}], Y[{mt_all[:, 1].min():.1f}, {mt_all[:, 1].max():.1f}]")
    
    if len(mt_converted) > 0:
        mt_conv_all = np.concatenate([vec for vec in mt_converted], axis=0)
        print(f"MapTracker Converted: X[{mt_conv_all[:, 0].min():.1f}, {mt_conv_all[:, 0].max():.1f}], Y[{mt_conv_all[:, 1].min():.1f}, {mt_conv_all[:, 1].max():.1f}]")
    
    if len(mr_vectors) > 0:
        mr_all = np.concatenate([vec for vec in mr_vectors], axis=0)
        print(f"MapTR Original:       X[{mr_all[:, 0].min():.1f}, {mr_all[:, 0].max():.1f}], Y[{mr_all[:, 1].min():.1f}, {mr_all[:, 1].max():.1f}]")

def main():
    parser = argparse.ArgumentParser(description='Visualize coordinate comparison between MapTracker and MapTR')
    parser.add_argument('--output-dir', default='/home/czhu/thesis/code/thesis_cheng/prediction_comparisons',
                       help='Output directory for visualizations')
    parser.add_argument('--num-scenes', type=int, default=6,
                       help='Number of scenes to visualize')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Confidence threshold for filtering predictions')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for scene selection')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    mt_data, mr_data = load_data()
    
    # Find common scenes
    mt_tokens = set(mt_data['results'].keys())
    mr_tokens = set()
    for sample in mr_data['results']:
        mr_tokens.add(sample['sample_token'])
    
    common_tokens = mt_tokens & mr_tokens
    print(f"Found {len(common_tokens)} common scenes")
    
    # Select random scenes
    random.seed(args.random_seed)
    selected_tokens = random.sample(list(common_tokens), min(args.num_scenes, len(common_tokens)))
    
    print(f"Selected {len(selected_tokens)} scenes for visualization:")
    for token in selected_tokens:
        print(f"  {token}")
    
    # Generate visualizations
    results = []
    
    for i, token in enumerate(selected_tokens):
        print(f"\nProcessing scene {i+1}/{len(selected_tokens)}: {token}")
        
        # Extract data
        data = extract_sample_data(mt_data, mr_data, token, args.confidence_threshold)
        if data[0] is None:
            print(f"  Skipping - no matching data found")
            continue
        
        (mt_vectors, mt_converted, mr_vectors, 
         mt_label_names, mt_label_names_converted, mr_label_names,
         mt_scores, mr_scores, mt_labels_converted) = data
        
        # Skip if no vectors
        if len(mt_vectors) == 0 and len(mr_vectors) == 0:
            print(f"  Skipping - no vectors found")
            continue
        
        # Analyze coordinate ranges
        analyze_coordinate_ranges(mt_vectors, mt_converted, mr_vectors, token)
        
        # Create visualization
        output_path = create_coordinate_comparison_plot(
            token, mt_vectors, mt_converted, mr_vectors,
            mt_label_names, mt_label_names_converted, mr_label_names,
            mt_scores, mr_scores, args.output_dir
        )
        
        # Store results
        results.append({
            'sample_token': token,
            'maptracker_vectors': len(mt_vectors),
            'maptr_vectors': len(mr_vectors),
            'maptracker_avg_score': np.mean(mt_scores) if len(mt_scores) > 0 else 0,
            'maptr_avg_score': np.mean(mr_scores) if len(mr_scores) > 0 else 0,
            'output_path': str(output_path)
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("COORDINATE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if results:
        total_mt_vectors = sum(r['maptracker_vectors'] for r in results)
        total_mr_vectors = sum(r['maptr_vectors'] for r in results)
        avg_mt_score = np.mean([r['maptracker_avg_score'] for r in results if r['maptracker_avg_score'] > 0])
        avg_mr_score = np.mean([r['maptr_avg_score'] for r in results if r['maptr_avg_score'] > 0])
        
        print(f"Processed {len(results)} scenes")
        print(f"Total MapTracker vectors: {total_mt_vectors}")
        print(f"Total MapTR vectors: {total_mr_vectors}")
        print(f"Average MapTracker confidence: {avg_mt_score:.3f}")
        print(f"Average MapTR confidence: {avg_mr_score:.3f}")
        
        # Save results
        results_path = output_dir / 'coordinate_comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Visualizations saved to: {output_dir}")
    else:
        print("No visualizations were generated")

if __name__ == "__main__":
    main()