# driver-pose-system
> driver pose system

## Dataset
Uploaded on Kaggle, see [here](https://www.kaggle.com/datasets/hlf2001/driver-monitor-dataset) for details.

## Environment
- Train Lightweight-Openpose on Kaggle
- Train ST-GCN on PC with CPU (training on GPU is not recommend, even slower than CPU)

Kaggle Notebook is [here](https://www.kaggle.com/code/hlf2001/driver-monitor)

On PC, see dependences in Pipfile.

Install environment
```sh
pip install pipenv
pipenv install
```

See release to get models and video.

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
```sh
export video_path=$video_path
export model_load_path=$model_load_path
export class_model_load_path=$class_model_load_path
pipenv run python app.py
```
