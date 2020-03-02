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

#### unhashable type: 'numpy.ndarray' error in tensorflow
placeholder 변수의 이름과 input 변수의 이름이 같을 때 위의 에러가 발생한다.

#### tf.train.import\_meta\_graph
meta graph를 불러와서 graph(network)를 재생성한다. <br/>
meta graph : tensorflow graph 정보 (모든 variables, operations 등을 저장하고 있음) <br/>

그런데 ```saver = tf.train.import_meta_graph('my_test_model-1000.meta')``` 이렇게 호출하면 이 코드 이전에 정의된 graph에 my\_test\_model\-1000.meta에 저장되어 있는 graph가 이어 붙는 방식으로 동작한다. 따라서 tf.train.import_meta_graph 함수 호출 전에 tf.reset\_default\_graph()를 호출해서 graph를 초기화 해주는 것이 좋다.
<br/>

graph에 이전에 저장된 변수 값들을 복구하기 위해서 ```saver.restore(sess, tf.train.lastest_checkpoint('./'))```를 호출한다.


