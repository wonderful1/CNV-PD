tra=./test.class.txt
tes=./test.class.txt
cla=2
outdir=./mode
mkdir -p ./mode
python ../src/pytorch_CNN_Train.py $tra $tes $cla $outdir
