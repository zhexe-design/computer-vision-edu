from pathlib import Path
import shutil
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

# Папка data рядом с этим скриптом
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)


def copy_to_data(source, filename):
    """Copy downloaded file into the data folder."""

    destination = DATA / filename
    shutil.copy2(source, destination)

    print(f"[OK] {destination}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WEEK 4 DATASET DOWNLOADER")
    print("=" * 60)

    print(f"\nDestination: {DATA}")

    # --------------------------------------------------------
    # 1. ICP POINT CLOUDS
    # --------------------------------------------------------

    print("\n[1/3] Downloading ICP point clouds...")

    dataset = o3d.data.DemoICPPointClouds()

    for i, source in enumerate(dataset.paths[:3], start=1):
        copy_to_data(source, f"scan{i}.ply")

    # --------------------------------------------------------
    # 2. STANFORD BUNNY
    # --------------------------------------------------------

    print("\n[2/3] Downloading Stanford Bunny...")

    bunny = o3d.data.BunnyMesh()

    copy_to_data(
        bunny.path,
        "bunny_mesh.ply"
    )

    # --------------------------------------------------------
    # 3. REDWOOD LIVING ROOM
    # --------------------------------------------------------

    print("\n[3/3] Downloading Redwood Living Room...")

    living_room = o3d.data.LivingRoomPointClouds()

    redwood_folder = DATA / "redwood_livingroom"
    redwood_folder.mkdir(parents=True, exist_ok=True)

    for i, source in enumerate(living_room.paths, start=1):

        destination = redwood_folder / f"fragment_{i:02d}.ply"

        shutil.copy2(source, destination)

        print(f"[OK] {destination}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)

    print(f"\nAll files saved in:\n{DATA}")

    print("\nDownloaded:")
    print("- 3 ICP point cloud fragments")
    print("- Stanford Bunny mesh")
    print("- 57 Redwood fragments")

    print("\nFolder structure:")
    print("data/")
    print("  scan1.ply")
    print("  scan2.ply")
    print("  scan3.ply")
    print("  bunny_mesh.ply")
    print("  redwood_livingroom/")
    print("    fragment_01.ply ... fragment_57.ply")


if __name__ == "__main__":
    main()