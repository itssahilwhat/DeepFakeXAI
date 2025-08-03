#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('.')

from src.core.config import Config

def analyze_dataset():
    """Analyze dataset composition and balance"""
    print("📊 Analyzing Dataset Composition")
    print("="*50)
    
    # Load manifest
    manifest_path = Path('data/wacv_data/manifest.csv')
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return None
    
    df = pd.read_csv(manifest_path)
    print(f"✅ Loaded manifest with {len(df)} samples")
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Real samples: {len(df[df['label'] == 0])}")
    print(f"  Fake samples: {len(df[df['label'] == 1])}")
    
    # Split distribution
    print(f"\n📂 Split Distribution:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")
    
    # Supervision distribution
    print(f"\n🎯 Supervision Distribution:")
    supervision_counts = df['supervision'].value_counts()
    for supervision, count in supervision_counts.items():
        print(f"  {supervision}: {count} samples")
    
    # Domain distribution
    if 'domain' in df.columns:
        print(f"\n🌍 Domain Distribution:")
        domain_counts = df['domain'].value_counts()
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} samples")
    
    # Method distribution (for fake samples)
    if 'method' in df.columns:
        print(f"\n🔧 Fake Generation Methods:")
        fake_df = df[df['label'] == 1]
        method_counts = fake_df['method'].value_counts()
        for method, count in method_counts.items():
            print(f"  {method}: {count} samples")
    
    # Detailed analysis by split and supervision
    print(f"\n📊 Detailed Analysis by Split and Supervision:")
    pivot_table = pd.crosstab(df['split'], df['supervision'], values=df['label'], aggfunc='count')
    print(pivot_table)
    
    # Class balance analysis
    print(f"\n⚖️ Class Balance Analysis:")
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        real_count = len(split_df[split_df['label'] == 0])
        fake_count = len(split_df[split_df['label'] == 1])
        total = len(split_df)
        
        print(f"  {split.upper()}:")
        print(f"    Real: {real_count} ({real_count/total*100:.1f}%)")
        print(f"    Fake: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"    Ratio: {real_count/fake_count:.2f}:1" if fake_count > 0 else "    Ratio: N/A")
    
    # Supervision balance analysis
    print(f"\n🎯 Supervision Balance Analysis:")
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        supervision_counts = split_df['supervision'].value_counts()
        
        print(f"  {split.upper()}:")
        for supervision, count in supervision_counts.items():
            percentage = count / len(split_df) * 100
            print(f"    {supervision}: {count} ({percentage:.1f}%)")
    
    # Check for missing masks
    print(f"\n🔍 Mask Availability Analysis:")
    missing_masks = df[df['mask_path'].isna() | (df['mask_path'] == '')]
    print(f"  Samples without masks: {len(missing_masks)} ({len(missing_masks)/len(df)*100:.1f}%)")
    
    if len(missing_masks) > 0:
        print(f"  Missing masks by supervision type:")
        missing_by_supervision = missing_masks['supervision'].value_counts()
        for supervision, count in missing_by_supervision.items():
            print(f"    {supervision}: {count}")
    
    return df

def create_visualizations(df, output_dir):
    """Create visualizations of dataset composition"""
    print(f"\n📊 Creating visualizations...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Composition Analysis', fontsize=16, fontweight='bold')
    
    # 1. Split distribution
    split_counts = df['split'].value_counts()
    axes[0, 0].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Split Distribution')
    
    # 2. Label distribution
    label_counts = df['label'].value_counts()
    axes[0, 1].pie(label_counts.values, labels=['Real', 'Fake'], autopct='%1.1f%%')
    axes[0, 1].set_title('Label Distribution')
    
    # 3. Supervision distribution
    supervision_counts = df['supervision'].value_counts()
    axes[0, 2].bar(supervision_counts.index, supervision_counts.values)
    axes[0, 2].set_title('Supervision Distribution')
    axes[0, 2].set_ylabel('Count')
    
    # 4. Split vs Label heatmap
    split_label_pivot = pd.crosstab(df['split'], df['label'], values=df['label'], aggfunc='count')
    sns.heatmap(split_label_pivot, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Split vs Label')
    axes[1, 0].set_xlabel('Label (0=Real, 1=Fake)')
    
    # 5. Split vs Supervision heatmap
    split_supervision_pivot = pd.crosstab(df['split'], df['supervision'], values=df['supervision'], aggfunc='count')
    sns.heatmap(split_supervision_pivot, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1])
    axes[1, 1].set_title('Split vs Supervision')
    
    # 6. Domain distribution (if available)
    if 'domain' in df.columns:
        domain_counts = df['domain'].value_counts()
        axes[1, 2].bar(domain_counts.index, domain_counts.values)
        axes[1, 2].set_title('Domain Distribution')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        # Method distribution for fake samples
        fake_df = df[df['label'] == 1]
        if 'method' in fake_df.columns and len(fake_df) > 0:
            method_counts = fake_df['method'].value_counts()
            axes[1, 2].bar(method_counts.index, method_counts.values)
            axes[1, 2].set_title('Fake Generation Methods')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, 'No method data available', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Method Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_composition.png', dpi=300, bbox_inches='tight')
    print(f"  📊 Visualization saved to: {output_dir / 'dataset_composition.png'}")
    
    # Create detailed summary table
    create_summary_table(df, output_dir)

def create_summary_table(df, output_dir):
    """Create detailed summary table"""
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Samples',
        'Value': len(df),
        'Percentage': '100%'
    })
    
    # Label statistics
    real_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    summary_data.append({
        'Metric': 'Real Samples',
        'Value': real_count,
        'Percentage': f'{real_count/len(df)*100:.1f}%'
    })
    summary_data.append({
        'Metric': 'Fake Samples',
        'Value': fake_count,
        'Percentage': f'{fake_count/len(df)*100:.1f}%'
    })
    
    # Split statistics
    for split in ['train', 'valid', 'test']:
        split_count = len(df[df['split'] == split])
        summary_data.append({
            'Metric': f'{split.upper()} Split',
            'Value': split_count,
            'Percentage': f'{split_count/len(df)*100:.1f}%'
        })
    
    # Supervision statistics
    for supervision in df['supervision'].unique():
        supervision_count = len(df[df['supervision'] == supervision])
        summary_data.append({
            'Metric': f'Supervision {supervision}',
            'Value': supervision_count,
            'Percentage': f'{supervision_count/len(df)*100:.1f}%'
        })
    
    # Domain statistics (if available)
    if 'domain' in df.columns:
        for domain in df['domain'].unique():
            domain_count = len(df[df['domain'] == domain])
            summary_data.append({
                'Metric': f'Domain {domain}',
                'Value': domain_count,
                'Percentage': f'{domain_count/len(df)*100:.1f}%'
            })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    print(f"  📋 Summary table saved to: {output_dir / 'dataset_summary.csv'}")
    
    # Print summary
    print(f"\n📋 Dataset Summary:")
    print(summary_df.to_string(index=False))

def check_sampling_balance(df):
    """Check if the dataset is properly balanced for training"""
    print(f"\n⚖️ Sampling Balance Check:")
    
    train_df = df[df['split'] == 'train']
    
    # Check class balance
    real_count = len(train_df[train_df['label'] == 0])
    fake_count = len(train_df[train_df['label'] == 1])
    total = len(train_df)
    
    real_ratio = real_count / total
    fake_ratio = fake_count / total
    
    print(f"  Training set class balance:")
    print(f"    Real: {real_count} ({real_ratio*100:.1f}%)")
    print(f"    Fake: {fake_count} ({fake_ratio*100:.1f}%)")
    
    # Check if balance is within acceptable range (40-60%)
    balance_ok = 0.4 <= real_ratio <= 0.6 and 0.4 <= fake_ratio <= 0.6
    print(f"    Balance acceptable (40-60%): {'✅' if balance_ok else '❌'}")
    
    if not balance_ok:
        print(f"    ⚠️  Consider using WeightedRandomSampler for balanced training")
    
    # Check supervision distribution
    print(f"\n  Training set supervision distribution:")
    supervision_counts = train_df['supervision'].value_counts()
    for supervision, count in supervision_counts.items():
        percentage = count / len(train_df) * 100
        print(f"    {supervision}: {count} ({percentage:.1f}%)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze dataset composition and balance')
    parser.add_argument('--output', type=str, default='outputs/dataset_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🔍 Deepfake Dataset Analysis")
    print("="*50)
    
    # Analyze dataset
    df = analyze_dataset()
    
    if df is not None:
        # Check sampling balance
        check_sampling_balance(df)
        
        # Generate visualizations
        if not args.no_viz:
            create_visualizations(df, args.output)
        
        print(f"\n✅ Dataset analysis completed!")
        print(f"📁 Results saved to: {args.output}")
    else:
        print("❌ Dataset analysis failed!")

if __name__ == "__main__":
    main() 