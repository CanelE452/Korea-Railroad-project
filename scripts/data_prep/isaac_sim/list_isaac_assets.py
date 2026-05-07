"""
Isaac Sim 4.5 에셋 경로 탐색 유틸리티

Isaac Sim에서 사용 가능한 Nucleus/S3 에셋 디렉토리를 열거하여
실제 존재하는 Props, Environments 등의 USD 파일 경로를 출력합니다.

사용법:
    conda activate pallet-pose
    set OMNI_KIT_ACCEPT_EULA=YES
    python scripts/data_prep/list_isaac_assets.py

    # 특정 디렉토리 탐색
    python scripts/data_prep/list_isaac_assets.py --search_dir "Props"

    # 깊이 제한
    python scripts/data_prep/list_isaac_assets.py --max_depth 2
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--search_dir", type=str, default=None,
                    help="Assets root 아래 탐색할 하위 디렉토리 (예: Props, Environments)")
parser.add_argument("--max_depth", type=int, default=2,
                    help="디렉토리 탐색 최대 깊이")
parser.add_argument("--filter", type=str, default=None,
                    help="파일명 필터 (대소문자 무시, 부분 매칭)")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.client


def list_dir_recursive(url, max_depth=2, _depth=0, indent=""):
    """Nucleus/S3 디렉토리를 재귀적으로 열거."""
    results = []
    result, entries = omni.client.list(url)
    if result != omni.client.Result.OK:
        print(f"{indent}[ERROR] Cannot list: {url} (result={result})")
        return results

    dirs = []
    files = []
    for entry in entries:
        full = f"{url}/{entry.relative_path}"
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            dirs.append((entry.relative_path, full))
        else:
            files.append((entry.relative_path, full))

    for name, full in sorted(files):
        if full.endswith((".usd", ".usda", ".usdz", ".usdc")):
            if args.filter is None or args.filter.lower() in name.lower():
                print(f"{indent}  {name}")
                results.append(full)

    for name, full in sorted(dirs):
        print(f"{indent}[DIR] {name}/")
        if _depth < max_depth:
            results.extend(list_dir_recursive(full, max_depth, _depth + 1, indent + "  "))

    return results


# Assets root 가져오기
try:
    from omni.isaac.nucleus import get_assets_root_path
    assets_root = get_assets_root_path()
    print(f"Assets root: {assets_root}")
except Exception as e:
    print(f"get_assets_root_path() failed: {e}")
    print("Trying default S3 path...")
    assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
    print(f"Using: {assets_root}")

if assets_root is None:
    print("No assets root found. Exiting.")
    simulation_app.close()
    sys.exit(1)

# 탐색 시작
if args.search_dir:
    search_url = f"{assets_root}/{args.search_dir}"
else:
    search_url = assets_root

print(f"\nSearching: {search_url}")
print(f"Max depth: {args.max_depth}")
if args.filter:
    print(f"Filter: {args.filter}")
print("=" * 60)

all_usd = list_dir_recursive(search_url, max_depth=args.max_depth)

print("=" * 60)
print(f"Total USD files found: {len(all_usd)}")

if all_usd:
    print("\nFull paths (copy-paste ready):")
    for p in all_usd:
        print(f"  \"{p}\",")

simulation_app.close()
