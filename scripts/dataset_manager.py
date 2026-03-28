#!/usr/bin/env python3
"""
Dataset Manager - Easily switch between and process different dataset versions
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np

DATA_ROOT = Path("/data/hafeez/graphdata")
GRAPH_ROOT = Path.home() / "Graph"

DATASETS = {
    "rooms": {
        "path": DATA_ROOT / "rooms",
        "description": "Original dataset - 1 scene (baseline/testing)",
        "scenes": 1,
        "ready": True
    },
    "rooms_update": {
        "path": DATA_ROOT / "rooms_update", 
        "description": "Updated dataset - 1 scene (new version)",
        "scenes": 1,
        "ready": False
    },
    "roomsmoved": {
        "path": DATA_ROOT / "roomsmoved",
        "description": "Large dataset - 97+ scenes (full experiments)",
        "scenes": 97,
        "ready": False
    }
}

def check_dataset_status(dataset_name):
    """Check if dataset has all required files"""
    dataset_path = DATASETS[dataset_name]["path"]
    
    status = {
        "dataset.npy": (dataset_path / "dataset.npy").exists(),
        "meshes/": (dataset_path / "meshes").exists() and len(list((dataset_path / "meshes").glob("*.ply"))) > 0,
        "pointclouds/": (dataset_path / "pointclouds").exists() and len(list((dataset_path / "pointclouds").glob("*.txt"))) > 0,
        "z_pc.npy": (dataset_path / "z_pc.npy").exists(),
        "scenes/": False
    }
    
    xml_files = list(dataset_path.glob("*.xml"))
    scenes_dir = dataset_path / "scenes"
    
    if scenes_dir.exists() and len(list(scenes_dir.glob("*.xml"))) > 0:
        status["scenes/"] = True
    elif len(xml_files) > 0:
        status["scenes/"] = "needs_organization"
    
    return status

def print_dataset_status():
    """Print status of all datasets"""
    print("\n" + "="*70)
    print("DATASET STATUS")
    print("="*70)
    
    for name, info in DATASETS.items():
        status = check_dataset_status(name)
        
        print(f"\n[{name.upper()}]")
        print(f"   Description: {info['description']}")
        print(f"   Path: {info['path']}")
        print(f"   Expected scenes: {info['scenes']}")
        print(f"   Status:")
        
        for component, exists in status.items():
            if exists == True:
                symbol = "[OK]"
            elif exists == "needs_organization":
                symbol = "[WARN]"
            else:
                symbol = "[MISSING]"
            print(f"      {symbol} {component}")
        
        ready = (status["dataset.npy"] and status["meshes/"] and 
                status["pointclouds/"] and status["z_pc.npy"])
        
        if ready:
            print(f"   >> READY for experiments")
        else:
            print(f"   >> NEEDS PROCESSING")
            
            needs = []
            if not status["pointclouds/"]:
                needs.append("pointclouds (run: python dataset_manager.py --prepare " + name + ")")
            if not status["z_pc.npy"]:
                needs.append("z_pc.npy (run: python dataset_manager.py --prepare " + name + ")")
            if status["scenes/"] == "needs_organization":
                needs.append("organize XMLs (run: python dataset_manager.py --organize " + name + ")")
            
            if needs:
                print(f"   Next steps:")
                for i, need in enumerate(needs, 1):
                    print(f"      {i}. {need}")
    
    print("\n" + "="*70 + "\n")

def organize_xml_files(dataset_name):
    """Move XML files to scenes/ subdirectory"""
    dataset_path = DATASETS[dataset_name]["path"]
    scenes_dir = dataset_path / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    
    xml_files = list(dataset_path.glob("*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {dataset_path}")
        return
    
    print(f"Moving {len(xml_files)} XML files to {scenes_dir}/")
    for xml in xml_files:
        target = scenes_dir / xml.name
        xml.rename(target)
        print(f"  [OK] {xml.name}")
    
    print(f"[DONE] Organized {len(xml_files)} XML files")

def prepare_dataset(dataset_name):
    """Prepare a dataset for use"""
    if dataset_name not in DATASETS:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return
    
    print(f"\n{'='*70}")
    print(f"PREPARING DATASET: {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    status = check_dataset_status(dataset_name)
    
    if status["scenes/"] == "needs_organization":
        print("\n[1/3] Organizing XML scene files...")
        organize_xml_files(dataset_name)
    else:
        print("\n[1/3] XML files already organized [OK]")
    
    if not status["pointclouds/"]:
        print("\n[2/3] Generating point clouds...")
        print(f"This may take a while for {dataset_name} (~{DATASETS[dataset_name]['scenes']} meshes)")
        
        cmd = [
            sys.executable,
            str(GRAPH_ROOT / "scripts" / "run_conversion_graph.py"),
            "--dataset", dataset_name
        ]
        
        result = subprocess.run(cmd, cwd=GRAPH_ROOT)
        if result.returncode != 0:
            print("[ERROR] Point cloud generation failed")
            return
        print("[DONE] Point clouds generated")
    else:
        print("\n[2/3] Point clouds already exist [OK]")
    
    if not status["z_pc.npy"]:
        print("\n[3/3] Extracting Point-MAE features...")
        print("This will take several minutes...")
        
        cmd = [
            sys.executable,
            str(GRAPH_ROOT / "scripts" / "extract.py"),
            "--dataset", dataset_name
        ]
        
        result = subprocess.run(cmd, cwd=GRAPH_ROOT)
        if result.returncode != 0:
            print("[ERROR] Feature extraction failed")
            return
        print("[DONE] Features extracted")
    else:
        print("\n[3/3] Features already extracted [OK]")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] {dataset_name.upper()} IS READY!")
    print(f"{'='*70}\n")
    print("Next steps:")
    print(f"  Test:  python scripts/run_experiment.py --config configs/{dataset_name}.yaml")
    print(f"  Train: python train_genie.py --config configs/{dataset_name}.yaml")
    print()

def get_dataset_info(dataset_name):
    """Get detailed information about a dataset"""
    dataset_path = DATASETS[dataset_name]["path"]
    
    try:
        tx, rx, xml, freq, tab = np.load(dataset_path / "dataset.npy", allow_pickle=True)
        
        print(f"\n{'='*70}")
        print(f"DATASET INFO: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Path: {dataset_path}")
        print(f"\nData arrays:")
        print(f"  Samples: {len(tx)}")
        print(f"  TX positions shape: {tx.shape if hasattr(tx, 'shape') else f'{len(tx)} samples'}")
        print(f"  RX positions shape: {rx.shape if hasattr(rx, 'shape') else f'{len(rx)} samples'}")
        print(f"  Frequency: {freq} Hz ({freq/1e9:.2f} GHz)")
        print(f"  Target (tab) shape: {tab.shape if hasattr(tab, 'shape') else f'{len(tab)} samples'}")
        
        meshes = list((dataset_path / "meshes").glob("*.ply"))
        pointclouds = list((dataset_path / "pointclouds").glob("*.txt")) if (dataset_path / "pointclouds").exists() else []
        
        scenes_dir = dataset_path / "scenes"
        if scenes_dir.exists():
            xml_files = list(scenes_dir.glob("*.xml"))
        else:
            xml_files = list(dataset_path.glob("*.xml"))
        
        print(f"\nFiles:")
        print(f"  Meshes: {len(meshes)}")
        print(f"  Point clouds: {len(pointclouds)}")
        print(f"  XML scenes: {len(xml_files)}")
        
        if (dataset_path / "z_pc.npy").exists():
            z_pc = np.load(dataset_path / "z_pc.npy")
            print(f"  Point-MAE features (z_pc): {z_pc.shape}")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage multiple dataset versions")
    parser.add_argument("--list", "-l", action="store_true", help="List all available datasets")
    parser.add_argument("--status", "-s", action="store_true", help="Show status of all datasets")
    parser.add_argument("--info", type=str, metavar="DATASET", help="Show detailed info about a dataset")
    parser.add_argument("--prepare", type=str, metavar="DATASET", help="Prepare a dataset")
    parser.add_argument("--organize", type=str, metavar="DATASET", help="Organize XML files")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        for name, info in DATASETS.items():
            print(f"  * {name:15s} - {info['description']}")
        print()
    elif args.status:
        print_dataset_status()
    elif args.info:
        if args.info not in DATASETS:
            print(f"[ERROR] Unknown dataset: {args.info}")
        else:
            get_dataset_info(args.info)
    elif args.prepare:
        prepare_dataset(args.prepare)
    elif args.organize:
        if args.organize not in DATASETS:
            print(f"[ERROR] Unknown dataset: {args.organize}")
        else:
            organize_xml_files(args.organize)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
