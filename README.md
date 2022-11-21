# Flow Supervisor

This repository publish code for the paper "Semi-Supervised Learning of Optical Flow by Flow Supervisor", ECCV 2022.
If you need to cite our work, consider to use this:
```bibtex
@inproceedings{Im_2022_ECCV,
  author = {Im, Woobin and Lee, Sebin and Yoon, Sung-Eui},
  title={Semi-Supervised Learning of Optical Flow by Flow Supervisor},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```

## TensorFlow code

* Our main code is included in the main directory
* To run experiment, see `train.sh`
* Install packages in `requirements.txt`
* Before running the experiment, download datasets for optical flow learning.
  * Please read `data/path.py` and locate the files according to the locations. 

### Pretrained model

Three pretrained models can be downloaded from [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/iwbn_kaist_ac_kr/EdFzcFpUwttPvc7sr5imnNsB41FoZe1rwuR6DWbkko-x9w?e=4wIjEW).

1. Chairs+Things model (ckpts/baseline/things/)
2. C+T+S^u model (ckpts/semi/sintel/)
3. C+T+K^u model (ckpts/semi/kitti/)
4. C+T+K^u model w/o unsup (ckpts/semi/kitti2/)
5. C+T+VKITTI+K^u model (ckpts/semi/vkitti/)
6. C+T+DAVIS model (ckpts/semi/davis_ct/)
7. C+T+S+K+H+DAVIS model (ckpts/semi/davis_ctskh/)

Originally the files were located in `./ckpts/`, but you can put them anywhere you want!

Use command below to test
```sh
python evaluate.py ckpts/semi/sintel/ckpt-100000-weights --gpus 0 --dataset sintel --eval_iters 12
```

## Extract optical flow from an image sequence
You can extract optical flow samples using `extract_flow.py` code.
```sh
python extract_flow.py ckpts/semi/davis_ctskh/ckpt-1 --model_type raft-semi --gpus 0 --source_dirs samples/davis/frames --target_dirs samples/davis/
```

### How to read `.flo` file

Refer to this code section in `raft_utils/frame_utils.py`. The resulting matrix contains $H\times W \times 2$ tensor; the last dimension represent a displacement vector $(x, y)$.
```python
import numpy as np
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
```

## Dockerfile
You can build the Tensorflow environment with `Dockerfile`.
Use this command to build:
```sh
docker build . --tag flow-supervisor
```

You can run a docker container with:
```sh
docker run --rm --gpus all flow-supervisor
```

### Acknowledgment
We thank the authors of [RAFT](https://github.com/princeton-vl/RAFT), and [SMURF](https://github.com/google-research/google-research/tree/master/smurf), [Uflow](https://github.com/google-research/google-research/tree/master/uflow)
for their contribution to the field and our research;
our implementation is inspired by, or utilized parts of 
RAFT, SMURF (network and losses), and Uflow code, as credited in our code.

## PyTorch code

* Our pytorch code (for benchmark) is included in `pytorch` directory
* We modified RAFT and GMA code for our flow supervisor method.
* Install packages in `requirements.txt`
* In PyTorch code, we use the dataset code implemented in our TensorFlow code.
  * Please read `wb_data/path.py` and locate the files according to the locations. 
* See `train_semi.sh` to run experiments. 
  * Pretrained weights should be downloaded before running the experiment. Download from original repositories for [RAFT](https://github.com/princeton-vl/RAFT) and [GMA](https://github.com/zacjiang/GMA).

