


import sys
import traceback
from pathlib import Path

import torch
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data

# ———— 配置区 ————
CIF_DIR = Path(r"D:/电导率预测/randomized_cifs")
OUT_DIR = Path(r"D:/电导率预测/processed_graphs")
CUTOFF = 5.0    # Å，邻居截断半径
# ——————————

def get_site_Z(site) -> int:
    """
    从 site.species (Composition) 中选占据度最高的元素，返回其原子序数 Z。
    """
    # site.species 是 pymatgen.core.composition.Composition
    species_occu = dict(site.species)  # {Element: occupancy}
    if not species_occu:
        raise ValueError(f"No species found for site {site}")
    dominant_elem = max(species_occu.items(), key=lambda kv: kv[1])[0]
    return dominant_elem.Z

def cif_to_pyg(cif_path: Path, cutoff: float = CUTOFF) -> Data:
    """
    将单个 .cif 文件转换为 PyG Data
    """
    # 1) 读取结构
    struct = Structure.from_file(str(cif_path))

    # 2) 构建节点特征 x: 每个 site 的主导元素 Z
    zs = [get_site_Z(site) for site in struct]
    x  = torch.tensor(zs, dtype=torch.float).view(-1, 1)  # [N,1]

    # 3) 周期性邻居搜索，构造边
    edge_index = []
    edge_attr  = []
    for i, site in enumerate(struct):
        neighs = struct.get_neighbors(site, cutoff)
        for nb in neighs:
            j    = nb.index
            dist = nb.nn_distance
            # 无向图：双向记录
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([dist])
            edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)                 # [E, 1]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def main():
    if not CIF_DIR.is_dir():
        print(f"[Error] CIF 目录不存在：{CIF_DIR}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(CIF_DIR.glob("*.cif"))
    if not cif_files:
        print(f"[Warning] 未在 {CIF_DIR} 下找到 .cif 文件。")
        return

    print(f"Found {len(cif_files)} CIF files in {CIF_DIR}")

    success, failed = 0, 0
    for cif_path in cif_files:
        try:
            data = cif_to_pyg(cif_path, cutoff=CUTOFF)
            save_path = OUT_DIR / (cif_path.stem + ".pt")
            torch.save(data, str(save_path))
            success += 1
            print(f"[OK]    {cif_path.name} → {save_path.name}")
        except Exception as e:
            failed += 1
            print(f"[Failed] {cif_path.name}: {e}")
            traceback.print_exc()

    print("\n===== Summary =====")
    print(f"Total CIFs : {len(cif_files)}")
    print(f"Succeeded  : {success}")
    print(f"Failed     : {failed}")
    print(f"Processed graphs saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
