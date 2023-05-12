# driver-pose-system
> driver pose system

## UI
![截屏2023-05-12 10 13 09](https://github.com/hlf20010508/driver-monitor-system/assets/76218469/24b2333e-dee8-46c1-81b1-e60ca6c42170)

## Dataset
Uploaded on Kaggle, see [here](https://www.kaggle.com/datasets/hlf2001/driver-monitor-dataset) for details.

## Pretrained Models
See [release](https://github.com/hlf20010508/driver-monitor-system/releases/tag/Models) to get pretrained models.

## Environment
- Train Lightweight-Openpose on Kaggle
- Train ST-GCN on PC with CPU (training on GPU is not recommend, even slower than CPU)

Kaggle Notebook is [here](https://www.kaggle.com/code/hlf2001/driver-monitor)

On PC, see dependences in [Pipfile](https://github.com/hlf20010508/driver-monitor-system/blob/master/Pipfile).

Install environment
```sh
pip install pipenv
pipenv install
```

## Train Lightweight-Openpose Model
```sh 
export annotation_path=$annotation_path
export img_root_path=$img_root_path
export model_save_dir=$model_save_dir
pipenv run python train_body.py
```

## Train ST-GCN Model
```sh 
export annotation_path=$annotation_path
export model_save_dir=$model_save_dir
pipenv run python train_body_class.py
```

## Launch app
See [release](https://github.com/hlf20010508/driver-monitor-system/releases/tag/Video) to get video.
```sh
export video_path=$video_path
export model_load_path=$model_load_path
export class_model_load_path=$class_model_load_path
pipenv run python app.py
```
