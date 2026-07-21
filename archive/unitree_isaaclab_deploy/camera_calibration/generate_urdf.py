#!/usr/bin/env python3
"""
Generate Go2 URDF with D435i camera from calibration results.

Usage:
    python generate_urdf.py \
        --base_urdf /path/to/go2.urdf \
        --calibration calibration_result.json \
        --output go2_with_d435i.urdf
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List


def create_d435i_link() -> ET.Element:
    """Create the D435i link element."""
    link = ET.Element("link", name="d435i_link")
    
    # Visual
    visual = ET.SubElement(link, "visual")
    visual_origin = ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    visual_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geom, "box", size="0.09 0.025 0.025")
    visual_mat = ET.SubElement(visual, "material", name="d435i_silver")
    ET.SubElement(visual_mat, "color", rgba="0.75 0.75 0.75 1")
    
    # Collision
    collision = ET.SubElement(link, "collision")
    collision_origin = ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    collision_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(collision_geom, "box", size="0.09 0.025 0.025")
    
    # Inertial (D435i mass ~72g)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.072")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "inertia", 
                  ixx="1e-5", ixy="0", ixz="0",
                  iyy="1e-5", iyz="0", izz="1e-5")
    
    return link


def create_d435i_joint(xyz: List[float], rpy: List[float], parent: str = "base") -> ET.Element:
    """Create the D435i joint element from calibration results."""
    joint = ET.Element("joint", name="d435i_joint", type="fixed")
    
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link="d435i_link")
    
    xyz_str = f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
    rpy_str = f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"
    ET.SubElement(joint, "origin", xyz=xyz_str, rpy=rpy_str)
    
    return joint


def create_d435i_optical_frame() -> tuple:
    """Create the D435i optical frame link and joint.
    
    The optical frame follows ROS convention: z forward, x right, y down.
    Offset accounts for the depth sensor position within the D435i body.
    """
    # Depth optical frame link (massless)
    link = ET.Element("link", name="d435i_depth_optical_frame")
    
    # Joint from d435i_link to optical frame
    # Offset: depth sensor is ~17.5mm to the right of center
    joint = ET.Element("joint", name="d435i_depth_optical_joint", type="fixed")
    ET.SubElement(joint, "parent", link="d435i_link")
    ET.SubElement(joint, "child", link="d435i_depth_optical_frame")
    # Transform from camera body to optical frame
    # Rotate to align with optical convention (z forward)
    ET.SubElement(joint, "origin", xyz="0 0.0175 0", rpy="-1.5708 0 -1.5708")
    
    return link, joint


def add_camera_to_urdf(base_urdf_path: str, 
                       calibration_path: str,
                       output_path: str,
                       parent_link: str = "base"):
    """Add D435i camera to existing Go2 URDF."""
    
    # Load calibration results
    with open(calibration_path, 'r') as f:
        calib = json.load(f)
    
    xyz = calib["xyz"]
    rpy = calib["rpy"]  # Already in radians
    
    print(f"Loading base URDF: {base_urdf_path}")
    print(f"Calibration - xyz: {xyz}, rpy (rad): {rpy}")
    
    # Parse base URDF
    tree = ET.parse(base_urdf_path)
    root = tree.getroot()
    
    # Check if camera already exists
    existing_links = [link.get("name") for link in root.findall("link")]
    if "d435i_link" in existing_links:
        print("Warning: d435i_link already exists in URDF, will be replaced")
        for elem in root.findall("link[@name='d435i_link']"):
            root.remove(elem)
        for elem in root.findall("joint[@name='d435i_joint']"):
            root.remove(elem)
        for elem in root.findall("link[@name='d435i_depth_optical_frame']"):
            root.remove(elem)
        for elem in root.findall("joint[@name='d435i_depth_optical_joint']"):
            root.remove(elem)
    
    # Add D435i elements
    d435i_link = create_d435i_link()
    d435i_joint = create_d435i_joint(xyz, rpy, parent_link)
    optical_link, optical_joint = create_d435i_optical_frame()
    
    root.append(d435i_link)
    root.append(d435i_joint)
    root.append(optical_link)
    root.append(optical_joint)
    
    # Write output
    indent_xml(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    print(f"Generated URDF saved to: {output_path}")
    
    # Print summary
    print("\nAdded elements:")
    print(f"  - d435i_link (visual, collision, inertial)")
    print(f"  - d435i_joint (fixed, parent={parent_link})")
    print(f"  - d435i_depth_optical_frame")
    print(f"  - d435i_depth_optical_joint")


def indent_xml(elem, level=0):
    """Add indentation to XML elements for readability."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def create_default_calibration(output_path: str):
    """Create a default calibration file for testing."""
    default_calib = {
        "transform_matrix": [
            [1, 0, 0, 0.35],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.15],
            [0, 0, 0, 1]
        ],
        "xyz": [0.35, 0.0, 0.15],
        "rpy": [-0.349, 0.0, 0.0],  # -20 degrees pitch
        "rpy_degrees": [-20.0, 0.0, 0.0],
        "position_error_mm": 0.0,
        "num_valid_pairs": 0,
        "note": "Default calibration - replace with actual calibration results"
    }
    
    with open(output_path, 'w') as f:
        json.dump(default_calib, f, indent=2)
    
    print(f"Created default calibration file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate URDF with D435i camera")
    parser.add_argument("--base_urdf", type=str, required=True,
                       help="Path to base Go2 URDF file")
    parser.add_argument("--calibration", type=str, required=True,
                       help="Path to calibration result JSON")
    parser.add_argument("--output", type=str, required=True,
                       help="Output URDF file path")
    parser.add_argument("--parent_link", type=str, default="base",
                       help="Parent link name for camera joint")
    parser.add_argument("--create_default_calib", action="store_true",
                       help="Create a default calibration file")
    args = parser.parse_args()
    
    if args.create_default_calib:
        create_default_calibration(args.calibration)
    
    if not Path(args.calibration).exists():
        print(f"Calibration file not found: {args.calibration}")
        print("Use --create_default_calib to create a default calibration file")
        return
    
    add_camera_to_urdf(
        args.base_urdf,
        args.calibration,
        args.output,
        args.parent_link
    )


if __name__ == "__main__":
    main()
