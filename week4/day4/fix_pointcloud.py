from pathlib import Path
import open3d as o3d

DATA = Path(__file__).resolve().parent / "data"

print("=" * 60)
print("CONVERTING POINT CLOUDS TO PLY")
print("=" * 60)

# Получаем исходный набор Open3D
dataset = o3d.data.DemoICPPointClouds()

for i, source_path in enumerate(dataset.paths, start=1):

    print(f"\n[{i}] Processing: {source_path}")

    # Читаем в исходном формате, который определяет Open3D
    cloud = o3d.io.read_point_cloud(source_path)

    if cloud.is_empty():
        print("[ERROR] Could not load point cloud!")
        continue

    destination = DATA / f"scan{i}.ply"

    # Настоящая конвертация в PLY
    success = o3d.io.write_point_cloud(
        str(destination),
        cloud,
        write_ascii=False
    )

    if success:
        print(f"[OK] Saved: {destination}")
        print(f"     Points: {len(cloud.points)}")
    else:
        print("[ERROR] Failed to save PLY!")

print("\n" + "=" * 60)
print("CONVERSION FINISHED")
print("=" * 60)