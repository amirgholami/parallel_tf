# parallel_tf

See fc_models/synchronized_sgd.py for fully controlled sync:
  - barrier setup (to overcome tensorflow protocol)
  - random seed fixing
  - batch shuffle control
  - recovery_wait_secs
  
See customCifarInputs for sync cifar-10 training with multiply gpu. Similar to above.

See process_group experiment in each model's folder.
  
This software has been developed and is maintained by the PALLAS group  
at the ASPIRE Lab in the University of California, Berkeley.

More information please visit: 
http://aspire.eecs.berkeley.edu
