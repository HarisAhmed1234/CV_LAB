import subprocess
from pathlib import Path
import sys

# Input photos and output directory
image_dir = Path("images").resolve()
output_dir = Path("colmap_output").resolve()

database_path = output_dir / "database.db"
sparse_dir = output_dir / "sparse"
dense_dir = output_dir / "dense"

# Check input images
if not image_dir.is_dir():
    print("Error: Create an 'images' folder and add your photos.")
    sys.exit(1)

image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
image_files = [
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

if len(image_files) < 2:
    print("Error: Add at least two images of the same scene.")
    sys.exit(1)

output_dir.mkdir(parents=True, exist_ok=True)
sparse_dir.mkdir(parents=True, exist_ok=True)


def run_colmap(args):
    """Run one COLMAP command and stop if it fails."""
    command = ["colmap"] + args
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True)


try:
    # 1. Detect image features
    run_colmap([
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir)
    ])

    # 2. Match features between images
    run_colmap([
        "exhaustive_matcher",
        "--database_path", str(database_path)
    ])

    # 3. Estimate camera poses and create a sparse 3D model
    run_colmap([
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir)
    ])

    # Find the reconstructed sparse model
    models = sorted(
        p for p in sparse_dir.iterdir()
        if p.is_dir() and (p / "images.bin").exists()
    )

    if not models:
        raise RuntimeError(
            "COLMAP did not produce a sparse model. "
            "Try images with more overlap and visible details."
        )

    sparse_model = models[0]

    # 4. Export the sparse point cloud
    sparse_ply = output_dir / "sparse_point_cloud.ply"

    run_colmap([
        "model_converter",
        "--input_path", str(sparse_model),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY"
    ])

    # 5. Prepare images for dense reconstruction
    run_colmap([
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"
    ])

    # 6. Compute dense stereo depth maps
    run_colmap([
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ])

    # 7. Fuse depth maps into a dense point cloud
    dense_ply = output_dir / "dense_point_cloud.ply"

    run_colmap([
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply)
    ])

    # 8. Generate a mesh from the dense point cloud
    mesh_ply = output_dir / "3d_mesh.ply"

    run_colmap([
        "poisson_mesher",
        "--input_path", str(dense_ply),
        "--output_path", str(mesh_ply)
    ])

    print("\n3D reconstruction completed!")
    print("Sparse point cloud:", sparse_ply)
    print("Dense point cloud:", dense_ply)
    print("3D mesh:", mesh_ply)

except FileNotFoundError:
    print("Error: COLMAP was not found. Install it or add it to PATH.")

except subprocess.CalledProcessError as error:
    print("A COLMAP step failed with exit code:", error.returncode)

except RuntimeError as error:
    print("Reconstruction error:", error)
