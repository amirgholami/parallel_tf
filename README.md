# parallel_tf

See fc_models/synchronized_sgd.py for fully controlled sync:
  - barrier setup (to overcome tensorflow protocol)
  - random seed fixing
  - batch shuffle control
  - recovery_wait_secs
  
See customCifarInputs for sync cifar-10 training with multiply gpu. Similar to above.

See process_group experiment in each model's folder.

![Alt text](./sync_cifar10_naive_cnn.png?raw=true "Cifar-10 sync training")

![Alt text](./async_sync_group_conv.png?raw=true "Title")

Examplify: async training failed to converge to the depth of sync training
![Alt text](./sync_async_cifar10.png?raw=true "Title")

![Alt text](./convergence_depth.png?raw=true "Convergence level in detail")  
This software has been developed and is maintained by the PALLAS group  
at the ASPIRE Lab in the University of California, Berkeley.

More information please visit: 
http://aspire.eecs.berkeley.edu
