# CNV-PD: Predicting Copy Number Variation (CNV) by Deep learning


## Install
CNV-PD runs on python3 environment
```
sklearn  
matplotlib  
pysam  
pandas  
numpy  
pytorch  
torchvision  
```

## USAGE
### 1. Generate Pileup images of candidate CNV regions
```
Usage: Get_Pileup_Image.py [-h] [-b bed] [-bas bas_file] [-bs bin_size]
                           [-k step_size] [-o outdir]
                           BamFileName

positional arguments:
  BamFileName           Bamfile to be used here

optional arguments:
  -h, --help            show this help message and exit
  -b bed, --bed bed     path to a .bed file with locations of CNVs. The second
                        column should be CNV start, and the third column
                        should be CNV end (default: None)
  -bas bas_file, --bas_file bas_file
                        path to a .bas file (default: None)
  -bs bin_size, --bin-size bin_size
                        the bin size that each bin will represent, in bases
                        (default: 100)
  -k step_size, --step-size step_size
                        the step-size for taking sliding windows to create
                        bins (default: 50)
  -o outdir, --output outdir
                        the directory into which output files will go
                        (default: .)
```
for example:  
```bash
python src/Get_Pileup_Image.py -b tests/test.bed -bas tests/test.bas -o ./ tests/test.bam

``` 


### 2. Training the CNN model
```bash
tra=train.txt  #  train-data for mode training
tes=test.txt  # test-data for check the accuracy of mode
cla=2  # 2-calss
outdir=./mode # directory for output mode
python src/pytorch_CNN_Train.py $tra $tes $cla $outdir
```


### 3. Prediction using the generated model
```BASH
testtxt=./test.class.txt  # the candidate CNVs you wanted to predicted
title='test_out' # prefix of output
class_num=2 # number of class
CNN=./CNN.2.model.pkl # the generated model you trained 
outdir=./ # directory for output file
python src/pytorch_CNN_Pre.py $testtxt $title $class_num $CNN $outdir
```

