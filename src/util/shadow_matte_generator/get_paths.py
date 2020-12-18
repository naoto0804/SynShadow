import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('amass_root', type=str,
                    default="data/3D_human/human_meshes_processed")
parser.add_argument('shapenet_root', type=str, default="data/ShapeNetCore.v2")
args = parser.parse_args()

# from amass
with Path("human_mesh_paths.txt").open(mode='w') as fd:
    for name in Path(args.amass_root).glob("*.stl"):
        fd.write(f"{str(Path.cwd() / name)}\n")

# from ShapeNet
with Path("shapecategories_outdoor.csv").open(mode='r') as fd:
    object_ids = [r[0] for r in csv.reader(fd)]
with Path("object_mesh_paths.txt").open(mode='w') as fd:
    name = "models/model_normalized.obj"
    for id_ in object_ids:
        for f in (Path(args.object_mesh_root) / id_).glob(f'*/{name}'):
            fd.write(f"{str(f)}\n")
