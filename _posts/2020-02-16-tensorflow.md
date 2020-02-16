---
layout: post
title: tensorflow 사용법
comments : true
category : Machine Learning
---

#### tf.train.Saver() 
```
__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```
var_list가 None이면 저장 가능한 모든 Variable과 Object를 저장한다. 
<br/>

#### tf.global_variables 
```Variable()``` 이나 ```get_variable```로 선언된 모든 변수 리스트를 return한다. 

#### tf.train.get\_checkpoint\_state()
```
Args:
checkpoint_dir: The directory of checkpoints.
latest_filename: Optional name of the checkpoint file. Default to 'checkpoint'.
```
CheckpointState를 return한다. <br/>
CheckpointState는 에는 두가지 정보가 담겨있다.
- model_checkpoint_path : 가장 최근에 저장된 job.ckpt 파일의 path 정보
- all_model_checkpoint_paths : 최근에 저장된 job_i.ckpt 파일들의 path 정보 list

#### tf.keras.backend.clear_session()
위 함수는 tf.Session이나 tf.InteractiveSession이 활성화 되어있을 때 호출하면 안된다.

#### colab에서 tensorboard 간편하게 사용하는 방법
%load_ext tensorboard <br/>
%tensorboard --logdir logs

#### tensorflow paddding
padding = 'SAME' : output_spacial_shape[i] = ceil(input_spatial_shape[i] / strides[i]) <br/>
padding = 'VALID' : output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]) <br/>
패딩을 추가하지 않는다. 따라서 윈도우가 움직이다가 남은 입력의 일부분이 윈도우 보다 작으면 해당 부분을 버린다. 
(ceil : 올림)

