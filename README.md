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

Three pretrained models can be downloaded from [here](#) (to be updated).

1. Chairs+Things model (ckpts/baseline/things/)
2. C+T+S^u model (ckpts/semi/sintel/)
3. C+T+K^u model (ckpts/semi/kitti/)

Originally the files were located in `./ckpts/`, but you can put them anywhere you want!

Use command below to test
```sh
python evaluate.py ckpts/semi/sintel/ckpt-100000-weights --gpus 0 --dataset sintel --eval_iters 12
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

