# 4DCT-OpFlow
4D-CT Dicom Images Optical Flow Neural Network

## Train without cost unrolling
- Branch out from `gpu` branch
- Run the following command:
  ```
  ~/4DCT-OpFlow/
  ❯ python main_train.py -c configs/server.json
  ```
- Run from checkpoint:
  ```
  ~/4DCT-OpFlow/
  ❯ python main_train.py -c configs/server.json -l checkpoints/some_checkpoint.tar.gz
  ```
  
## Train with cost unrolling
- Branch out from `cost_unrolling_gpu` branch
- Run the following command:
  ```
  ~/4DCT-OpFlow/
  ❯ python main_train.py -c configs/server_admm_v2.json
  ```
- Run from checkpoint:
  ```
  ~/4DCT-OpFlow/
  ❯ python main_train.py -c configs/server_admm_v2.json -l checkpoints/some_checkpoint.tar.gz
  ```
## For more info on training
 ```
  ~/4DCT-OpFlow/
  ❯ python main_train.py --help
 ```
