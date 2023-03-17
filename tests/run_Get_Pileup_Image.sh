bam=./test.bam
bed=./test.bed
bas=./test.bam.bas
python ../src/Get_Pileup_Image.py -b $bed -bas $bas -o . $bam
