# driver-pose-estimation
> driver pose estimation

```sh 
export base_model=mnv3s
export annotation_path=$annotation_path
export img_root_path=$img_root_path
export model_save_dir=$model_save_dir
pipenv run python train_body.py
```

```sh 
export base_model=mnv3s
export annotation_path=$annotation_path
export img_root_path=$img_root_path
export model_save_dir=$model_save_dir
pipenv run python train_face.py
```