"""
WEEK 3 — DAY 6
Point Cloud Project Finalization and ICP Readiness Check

Features:
1. Validate project structure.
2. Check Python scripts and output files.
3. Inspect saved point clouds using Open3D.
4. Read parameters from Day 5 configuration.
5. Collect portfolio-ready visualizations.
6. Generate a project audit report.
7. Check readiness for the next topic: ICP Registration.

Requirements:
    pip install open3d numpy

Run from repository root:
    python week3_pointclouds/day6_finalize.py
"""

from pathlib import Path
from datetime import datetime
import re
import shutil
import sys

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

CLEANED_DIR = RESULTS_DIR / "cleaned"
NORMALS_DIR = RESULTS_DIR / "with_normals"
CROPPED_DIR = RESULTS_DIR / "cropped"

VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
SUMMARY_DIR = RESULTS_DIR / "summaries"

PORTFOLIO_DIR = RESULTS_DIR / "portfolio"

AUDIT_PATH = RESULTS_DIR / "day6_audit_report.txt"

DAY5_SCRIPT = PROJECT_DIR / "day5_full_pipeline.py"

REQUIRED_SCRIPTS = [
    "day1_pointcloud_basics.py",
    "day2_filtering.py",
    "day3_normals_and_roi.py",
    "day4_advanced_cleaning.py",
    "day5_full_pipeline.py",
]

EXPECTED_RESULT_FILES = {
    "Cleaned point cloud": (
        CLEANED_DIR / "cleaned_pointcloud.ply"
    ),
    "Point cloud with normals": (
        NORMALS_DIR / "pointcloud_with_normals.ply"
    ),
    "Cropped ROI": (
        CROPPED_DIR / "cropped_roi.ply"
    ),
}

EXPECTED_VISUALIZATIONS = {
    "Original point cloud": (
        VISUALIZATION_DIR / "01_original.png"
    ),
    "Cleaned point cloud": (
        VISUALIZATION_DIR / "02_cleaned.png"
    ),
    "Point cloud with normals": (
        VISUALIZATION_DIR / "03_with_normals.png"
    ),
    "Cropped ROI": (
        VISUALIZATION_DIR / "04_cropped_roi.png"
    ),
}

SUPPORTED_POINT_CLOUD_FORMATS = {
    ".ply",
    ".pcd",
    ".xyz",
    ".xyzn",
    ".xyzrgb",
    ".pts",
}

TEMPORARY_EXTENSIONS = {
    ".tmp",
    ".bak",
    ".old",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_directories():
    """Create directories required for the final package."""

    for directory in [
        DATA_DIR,
        RESULTS_DIR,
        CLEANED_DIR,
        NORMALS_DIR,
        CROPPED_DIR,
        VISUALIZATION_DIR,
        SUMMARY_DIR,
        PORTFOLIO_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def format_size(size_bytes):
    """Format a file size in readable units."""

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


def record_result(results, name, passed, message):
    """Record a check and print its result."""

    status = "PASS" if passed else "FAIL"

    results.append({
        "name": name,
        "passed": passed,
        "message": message,
    })

    print(f"[{status}] {name}: {message}")

def download_dataset_if_missing():
    """Download Stanford Bunny if no point cloud exists in data/."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_POINT_CLOUD_FORMATS
    ]

    if existing_files:
        print("[INFO] Dataset already exists:")
        for path in existing_files:
            print(f"  {path.name}")
        return

    print("[INFO] Downloading Stanford Bunny dataset...")

    try:
        dataset = o3d.data.BunnyMesh()
        mesh = o3d.io.read_triangle_mesh(dataset.path)

        if mesh.is_empty():
            raise RuntimeError("Downloaded mesh is empty.")

        point_cloud = mesh.sample_points_uniformly(
            number_of_points=30000
        )

        output_path = DATA_DIR / "bunny.ply"

        success = o3d.io.write_point_cloud(
            str(output_path),
            point_cloud
        )

        if not success:
            raise RuntimeError("Could not save the point cloud.")

        print(f"[SUCCESS] Dataset saved to: {output_path}")
        print(f"[INFO] Number of points: {len(point_cloud.points):,}")

    except Exception as error:
        print(f"[ERROR] Dataset download failed: {error}")
        raise

# ============================================================
# PROJECT STRUCTURE CHECK
# ============================================================

def check_project_structure():
    """Verify the expected project structure."""

    print_header("STEP 1 — PROJECT STRUCTURE")

    results = []

    for script_name in REQUIRED_SCRIPTS:
        script_path = PROJECT_DIR / script_name

        record_result(
            results,
            script_name,
            script_path.is_file(),
            (
                f"Found: {script_path}"
                if script_path.is_file()
                else "Missing script"
            ),
        )

    required_directories = [
        DATA_DIR,
        RESULTS_DIR,
        CLEANED_DIR,
        NORMALS_DIR,
        CROPPED_DIR,
    ]

    for directory in required_directories:
        record_result(
            results,
            str(directory.relative_to(PROJECT_DIR)),
            directory.is_dir(),
            "Directory exists" if directory.is_dir()
            else "Directory is missing",
        )

    return results


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset():
    """List source point clouds and inspect the available data."""

    print_header("STEP 2 — INPUT DATA")

    files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_POINT_CLOUD_FORMATS
    )

    if not files:
        print("[WARNING] No point cloud files found in data/.")
        return []

    for path in files:
        print(f"{path.name:<40} {format_size(path.stat().st_size)}")

    return files


# ============================================================
# DAY 5 PARAMETER EXTRACTION
# ============================================================

def extract_parameter(source_code, variable_name, default):
    """
    Extract a simple constant assignment from the Day 5 script.

    This handles assignments such as:
        VOXEL_SIZE = 0.01
        USE_RADIUS_FILTER = True

    If the value cannot be parsed, return the supplied default.
    """

    pattern = (
        rf"^\s*{re.escape(variable_name)}\s*=\s*(.*?)\s*$"
    )

    match = re.search(
        pattern,
        source_code,
        flags=re.MULTILINE,
    )

    if not match:
        return default

    value = match.group(1).split("#", 1)[0].strip()

    if not value:
        return default

    return value


def inspect_day5_parameters():
    """Read configured cleaning parameters from the Day 5 script."""

    print_header("STEP 3 — DAY 5 PARAMETERS")

    defaults = {
        "VOXEL_SIZE": "0.01",
        "STATISTICAL_NB_NEIGHBORS": "20",
        "STATISTICAL_STD_RATIO": "2.0",
        "USE_RADIUS_FILTER": "True",
        "RADIUS_NB_POINTS": "8",
        "RADIUS_SIZE": "0.025",
        "NORMAL_RADIUS": "0.075",
        "NORMAL_MAX_NN": "30",
        "ROI_MIN_RATIO": "0.10",
        "ROI_MAX_RATIO": "0.90",
    }

    if not DAY5_SCRIPT.is_file():
        print("[WARNING] Day 5 script is missing.")
        return defaults

    source_code = DAY5_SCRIPT.read_text(
        encoding="utf-8",
        errors="replace",
    )

    parameters = {}

    for name, default in defaults.items():
        parameters[name] = extract_parameter(
            source_code,
            name,
            default,
        )

        print(f"{name:<32} {parameters[name]}")

    return parameters


# ============================================================
# POINT CLOUD VALIDATION
# ============================================================

def validate_point_cloud(path):
    """
    Check that a saved point cloud is readable and nonempty.

    For clouds with normals, verify that normals exist and
    correspond to the number of points.
    """

    if not path.is_file():
        return False, "File does not exist"

    try:
        cloud = o3d.io.read_point_cloud(str(path))

        if cloud.is_empty():
            return False, "Cloud is empty or unreadable"

        points = np.asarray(cloud.points)

        if not np.isfinite(points).all():
            return False, "Cloud contains non-finite coordinates"

        details = [
            f"{len(points):,} points",
            f"file size {format_size(path.stat().st_size)}",
        ]

        if "normals" in path.name.lower():
            if not cloud.has_normals():
                return False, (
                    f"Loaded {len(points):,} points, "
                    "but normals are missing"
                )

            normals = np.asarray(cloud.normals)

            if len(normals) != len(points):
                return False, "Normal count does not match point count"

            if not np.isfinite(normals).all():
                return False, "Normals contain non-finite values"

            normal_lengths = np.linalg.norm(normals, axis=1)
            valid_normals = normal_lengths > 1e-12

            details.append(
                f"valid normals {valid_normals.sum():,}/{len(normals):,}"
            )

            if not valid_normals.any():
                return False, "All normals have zero length"

        return True, "; ".join(details)

    except Exception as error:
        return False, f"Could not read cloud: {error}"


def check_final_results():
    """Check all expected final point cloud files."""

    print_header("STEP 4 — FINAL POINT CLOUDS")

    results = []

    for label, path in EXPECTED_RESULT_FILES.items():
        passed, message = validate_point_cloud(path)

        record_result(
            results,
            label,
            passed,
            message,
        )

    return results


# ============================================================
# VISUALIZATION CHECK
# ============================================================

def check_visualizations():
    """Check and collect available PNG visualizations."""

    print_header("STEP 5 — VISUALIZATIONS")

    results = []

    for label, path in EXPECTED_VISUALIZATIONS.items():
        exists = path.is_file()

        if exists:
            message = format_size(path.stat().st_size)

            if path.stat().st_size == 0:
                exists = False
                message = "File is empty"
        else:
            message = "Visualization is missing"

        record_result(
            results,
            label,
            exists,
            message,
        )

    return results


def collect_portfolio_assets():
    """
    Copy available final screenshots into one portfolio folder.

    Source files remain untouched.
    """

    print_header("STEP 6 — PORTFOLIO ASSETS")

    copied_files = []

    for label, source in EXPECTED_VISUALIZATIONS.items():
        if not source.is_file():
            continue

        if source.stat().st_size == 0:
            continue

        destination = PORTFOLIO_DIR / source.name

        try:
            shutil.copy2(source, destination)
            copied_files.append(destination)

            print(
                f"[COPIED] {source.name} -> "
                f"{destination.relative_to(PROJECT_DIR)}"
            )

        except OSError as error:
            print(f"[WARNING] Could not copy {source.name}: {error}")

    if not copied_files:
        print(
            "[WARNING] No visualizations were collected. "
            "Run Day 5 first."
        )

    return copied_files


# ============================================================
# TEMPORARY FILE REVIEW
# ============================================================

def find_temporary_files():
    """
    Find obvious temporary files for manual review.

    Nothing is deleted. Dataset files and results are preserved.
    """

    print_header("STEP 7 — TEMPORARY FILE REVIEW")

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
    }

    temporary_files = []

    for path in PROJECT_DIR.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(PROJECT_DIR)

        if any(part in ignored_directories for part in relative.parts):
            continue

        if path.suffix.lower() in TEMPORARY_EXTENSIONS:
            temporary_files.append(path)

        elif path.name.startswith("~$"):
            temporary_files.append(path)

    if not temporary_files:
        print("No obvious temporary files found.")
    else:
        print(
            "Review the following files manually. "
            "This script will not delete them:"
        )

        for path in temporary_files:
            print(f"  {path.relative_to(PROJECT_DIR)}")

    return temporary_files


# README generation intentionally omitted.


# ============================================================
# ICP READINESS CHECK
# ============================================================

def check_icp_readiness():
    """
    Perform a basic ICP preparation check.

    This does not run ICP or guarantee that two clouds can
    be successfully registered.
    """

    print_header("STEP 9 — ICP READINESS")

    results = []

    cleaned_path = EXPECTED_RESULT_FILES["Cleaned point cloud"]
    normals_path = EXPECTED_RESULT_FILES["Point cloud with normals"]

    cleaned_ok, cleaned_message = validate_point_cloud(cleaned_path)

    record_result(
        results,
        "Cleaned source cloud",
        cleaned_ok,
        cleaned_message,
    )

    normals_ok, normals_message = validate_point_cloud(normals_path)

    record_result(
        results,
        "Cloud with normals",
        normals_ok,
        normals_message,
    )

    if cleaned_ok and normals_ok:
        cleaned = o3d.io.read_point_cloud(str(cleaned_path))
        with_normals = o3d.io.read_point_cloud(str(normals_path))

        same_count = (
            len(cleaned.points) == len(with_normals.points)
        )

        record_result(
            results,
            "Point count consistency",
            same_count,
            (
                f"{len(cleaned.points):,} vs "
                f"{len(with_normals.points):,} points"
            ),
        )

    else:
        record_result(
            results,
            "Point count consistency",
            False,
            "Cannot compare clouds because one or both are invalid",
        )

    print(
        "\nReminder: ICP requires at least two point clouds, "
        "usually a source and a target."
    )

    print(
        "This check verifies preprocessing outputs only. "
        "It does not test registration accuracy."
    )

    return results


# ============================================================
# FINAL AUDIT REPORT
# ============================================================

def write_audit_report(
    structure_results,
    cloud_results,
    visualization_results,
    icp_results,
    temporary_files,
):
    """Write a project audit report without generating a README."""

    all_results = (
        structure_results
        + cloud_results
        + visualization_results
        + icp_results
    )

    passed = sum(result["passed"] for result in all_results)
    failed = len(all_results) - passed

    lines = [
        "WEEK 3 — DAY 6",
        "PROJECT FINALIZATION AUDIT",
        "=" * 70,
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Project directory: {PROJECT_DIR}",
        f"Checks passed: {passed}",
        f"Checks failed: {failed}",
        "",
        "CHECK DETAILS",
        "-" * 70,
    ]

    for result in all_results:
        status = "PASS" if result["passed"] else "FAIL"

        lines.append(
            f"[{status}] {result['name']}: {result['message']}"
        )

    lines.extend([
        "",
        "",
        "TEMPORARY FILES FOR MANUAL REVIEW",
        "-" * 70,
    ])

    if temporary_files:
        for path in temporary_files:
            lines.append(str(path.relative_to(PROJECT_DIR)))
    else:
        lines.append("No obvious temporary files detected.")

    lines.extend([
        "",
        "FINAL MANUAL CHECKLIST",
        "-" * 70,
        "[ ] Confirm dataset source and licensing.",
        "[ ] Inspect all visualizations.",
        "[ ] Verify final cleaning parameters.",
        "[ ] Remove temporary files manually if appropriate.",
        "[ ] Run the main Day 5 script from a clean checkout.",
        "[ ] Commit only intended source files and final results.",
        "[ ] Prepare source and target clouds for ICP.",
        "",
        "IMPORTANT",
        "-" * 70,
        "This report does not certify scientific correctness.",
        "Visual inspection and manual review are still required.",
        "",
    ])

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[SAVED] {AUDIT_PATH}")

    print_header("FINAL SCORE")

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")

    if failed == 0:
        print("All automated checks passed.")
        print("Complete the manual checklist before publishing.")
    else:
        print("Resolve the failed checks before publishing.")

    print("\nThis score covers automated checks only.")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete finalization and audit workflow."""

    print_header("WEEK 3 — DAY 6: FINAL PROJECT PACKAGING")

    print(f"Project: {PROJECT_DIR}")

    create_directories()

    download_dataset_if_missing()

    structure_results = check_project_structure()

    dataset_files = inspect_dataset()

    parameters = inspect_day5_parameters()

    cloud_results = check_final_results()

    visualization_results = check_visualizations()

    collect_portfolio_assets()

    temporary_files = find_temporary_files()

    icp_results = check_icp_readiness()

    write_audit_report(
        structure_results=structure_results,
        cloud_results=cloud_results,
        visualization_results=visualization_results,
        icp_results=icp_results,
        temporary_files=temporary_files,
    )

    print_header("WEEK 3 — DAY 6 COMPLETED")

    print("Review the audit report before publishing.")
    print("No project files were automatically deleted.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n[STOPPED] Execution interrupted by user.")
        sys.exit(130)

    except Exception as error:
        print(f"\n[UNEXPECTED ERROR] {error}")
        raise