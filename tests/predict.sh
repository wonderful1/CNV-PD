testtxt=./test.class.txt
title='test_out'
class_num=2
CNN=./CNN.2.model.pkl
outdir=./

python ../src/pytorch_CNN_Pre.py $testtxt $title $class_num $CNN $outdir
