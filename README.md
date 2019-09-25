# CAT
Implementation of CAT paper


This code implements the papper "CAT - Compression-Aware Training  for bandwidth reduction"

Arxiv link - TODO


## Datasets  
  
To run this code you need the training and validation set of ILSVRC2012 data

To get the ILSVRC2012 data, you should register on their [site](http://www.image-net.org/download-imageurls) for access
   

## Running instructions

python main.py --data <ILSVRC2012 folder location> --model <Model name (resnet18 / resnet34 / resnet50 / mobilenet_v2)>  --actBitwidth <Bits for main principal component> --weightBitwidth <8/32>  --clip --method <compression \ entropy> --regul <Lambda value>



## Results
 ![results](imgs/results1.PNG)
 
 ![results](imgs/Comparison.PNG)
  
 ![results](imgs/ablation.PNG)
 
  
## Acknowledgments  
TODO

## Citation  
TODO  
