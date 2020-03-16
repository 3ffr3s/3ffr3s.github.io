---
layout: post
title: "Universal Adversarial Perturbation 구현하기"
comments : true
category : Machine Learning
---

Universal Adversarial Perturbation 생성 알고리즘을 텐서플로우 1.X를 이용해서 구현해보았다. 
<br/>
코드는 https://github.com/LTS4/universal를 참조했다. 
<br/>
(저기에 있는 코드를 거의 그대로 따라했다.)

## Universal Adversarial Perturbation 
```python
def proj_lp(v, xi, p):
  if p == 2:
    v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
  
  elif p == np.inf:
    v = np.sign(v) * np.minimum(abs(v), xi)
  
  else:
      raise ValueError('Values of p different from 2 and inf are currently not supported') 

  return v
  
def universal_pert(dataset, f, grads, delta = 0.2, max_iter_uni = np.inf, xi = 10 , p =np.inf, num_classes = 10, overshoot = 0.02, max_iter_df = 50):

  image_shape = dataset[0].shape

  v = np.zeros(image_shape)
  fooling_rate = 0.0
  num_images = dataset.shape[0]

  itr = 0

  while fooling_rate <= (1-delta) and itr < max_iter_uni:
    np.random.shuffle(dataset)

    for i in range(num_images):

      cur_image = dataset[i]      
      pert_image = cur_image + v

      if np.argmax(f(cur_image).flatten()) == np.argmax(f(pert_image).flatten()):

        dr, loop_i, _, _ = DeepFool(pert_image, f, grads, num_classes, overshoot, max_iter_df)

        if loop_i <= max_iter_df -1:
          v = v + dr
          v = proj_lp(v,xi,p)
        
    itr = itr + 1

    pert_dataset = dataset + v

    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    batch_size = 100

    num_batches = np.int(np.ceil( np.float(num_images) / np.float(batch_size)))

    for ii in range(0,num_batches):
      m = (ii * batch_size)
      M = min((ii+1) * batch_size, num_images)

      est_labels_orig[m:M] = np.argmax(f(dataset[m:M,:,:,:].squeeze(axis = 1)), axis = 1)
      est_labels_pert[m:M] = np.argmax(f(pert_dataset[m:M,:,:,:].squeeze(axis =1 )), axis = 1 )

    fooling_rate = 1.0 - np.mean(np.equal(est_labels_orig, est_labels_pert).astype('float32'))

    print('fooling_rate = ', fooling_rate)

  return v
```

