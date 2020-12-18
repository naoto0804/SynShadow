import argparse
from pathlib import Path

import numpy as np
import torch

import trimesh
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import colors
from human_body_prior.tools.omni_tools import copy2cpu as c2c

parser = argparse.ArgumentParser()
parser.add_argument('--input_root', type=str, default="data/3D_human")
parser.add_argument('--identifier', type=str, default="human_meshes_processed")
args = parser.parse_args()

np.random.seed(0)
NUM_BETAS = 10  # number of body parameters
NUM_DMPLS = 8  # number of DMPL parameters

# Choose the device to run the body model on.
comp_device = torch.device('cpu')

input_root = Path(args.input_root)
output_root = Path(args.input_root) / args.identifier
output_root.mkdir(parents=True, exist_ok=True)
assert input_root.exists()

id_path_list = []
cnt = 0

model_types = ['male', 'female', 'neutral']

bm_dict, faces_dict = {}, {}
for m in model_types:
    bm_path = str(input_root / 'MANO/smplh' / m / 'model.npz')
    dmpl_path = str(input_root / 'DMPL' / m / 'model.npz')
    bm_dict[m] = BodyModel(
        bm_path=bm_path, num_betas=NUM_BETAS,
        num_dmpls=NUM_DMPLS, path_dmpl=dmpl_path).to(comp_device)
    faces_dict[m] = c2c(bm_dict[m].f)


def process(npz_bdata_path: Path):
    global cnt

    print(cnt, npz_bdata_path)
    bdata = np.load(str(npz_bdata_path))
    # print('Data keys available:%s' % list(bdata.keys()))
    # beta means shape
    # num of elements: 156 (pose), 8 (dmpl), 16 (beta)
    # print('Vector poses has %d elements for each of %d frames.' %
    #       (bdata['poses'].shape[1], bdata['poses'].shape[0]))
    # print('Vector dmpls has %d elements for each of %d frames.' %
    #       (bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
    # print('Vector trams has %d elements for each of %d frames.' %
    #       (bdata['trans'].shape[1], bdata['trans'].shape[0]))
    # print('Vector betas has %d elements constant for the whole sequence.' %
    #       bdata['betas'].shape[0])
    # print('The subject of the mocap sequence is %s.' % bdata['gender'])

    try:
        # frame id of the mocap sequence
        fId = np.random.randint(len(bdata['poses']))
        # gender = str(bdata['gender'])

        # Don't know why, but there exists key like b'female'..
        if "female" in str(bdata['gender']):
            gender = "female"
        elif "male" in str(bdata['gender']):
            gender = "male"
        elif "neutral" in str(bdata['gender']):
            gender = "neutral"
        else:
            print(cnt, npz_bdata_path, str(bdata['gender']))
            raise NotImplementedError
        # controls the global root orientation
        # root_orient = torch.Tensor(
        # bdata['poses'][fId:fId+1, :3]).to(comp_device)
        # controls the body
        pose_body = torch.Tensor(
            bdata['poses'][fId:fId+1, 3:66]).to(comp_device)
        # controls the finger articulation
        pose_hand = torch.Tensor(
            bdata['poses'][fId:fId+1, 66:]).to(comp_device)
        # controls the body shape
        betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)
        # controls soft tissue dynamics
        dmpls = torch.Tensor(bdata['dmpls'][fId:fId+1]).to(comp_device)
        # trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)
    except KeyError:
        return

    output_stl_name = output_root / f'{cnt:06}.stl'
    if not output_stl_name.exists():
        # ignore only rotation/translation because it is easy to augment
        body = bm_dict[gender](
            pose_body=pose_body, betas=betas, pose_hand=pose_hand, dmpls=dmpls)
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body.v[0]), faces=faces_dict[gender],
            vertex_colors=np.tile(colors['grey'], (6890, 1)))
        body_mesh.export(output_stl_name)

    # record and increment
    id_path_list.append((f'{cnt:06}', f))
    cnt += 1


for f in (input_root / 'AMASS/body_data').glob('**/*.npz'):
    process(f)

# with (output_root.parent / f'{args.identifier}.csv').open(mode='w') as fd:
#     for (i, f) in id_path_list:
#         fd.write(f'{i},{f}\n')
