# Dependencies
- Blender
- Python3

# Preparation

1. AMASS: Download human mesh sequences from [AMASS](https://amass.is.tue.mpg.de/) and base body models from [MANO](https://mano.is.tue.mpg.de/) and [DMPL](https://psfiles.is.tuebingen.mpg.de/downloads/smpl/dmpls-tar-xz).

2. ShapeNet: Download ShapeNetCore.v2 from [ShapeNet](https://www.shapenet.org/) and put it under `data/`.

Files should be organized as follows:

```
- data
    - 3D_human
        - AMASS
        - MANO
        - DMPL
    - ShapeNetCore.v2
```

# Instruction

1. Install dependencies

```
$ bash setup.sh
```

2. render human meshes from original AMASS dataset

```
$ python get_human_meshes.py
```

3. Get paths for all the 3D models

```
$ python get_paths.py
```

4. Render shadow matte in .exr format

```
$ blender -b -P render.py
```

5. Convert .exr to .png

```
$ python postprocess.py
```
