import sys
import os

def Read_bed(bed,path,sam,soft):
	f2 = open("../"+sam+"/"+soft+"/Class.2.txt", "w")
	f4 = open("../"+sam+"/"+soft+"/Class.4.txt", "w")
	if not os.path.isfile(bed):
		print('Bedfile does not exist at location provided, please check (tried "{}")'.format(bed))
		exit(2)
	print("Loading bed file: {} ...".format(bed))
	bed_f = open(bed)
	CNV_bed=[]
	i=0
	for line in bed_f:
		line=line.strip("\n")
		line_list=line.split("\t")
		#CNV_bed.append(line_list)
		if line_list[5]=="1" : Cla1="1"
		else: Cla1="0"

		if line_list[4]=="0" and line_list[5]=="0": Cla2="0"
		if line_list[4]=="0" and line_list[5]=="1": Cla2="1"
		if line_list[4]=="1" and line_list[5]=="0": Cla2="2"
		if line_list[4]=="1" and line_list[5]=="1": Cla2="3"

		f2.write(path+"/"+sam+"/"+soft+"/Pileup_Image/"+str(i)+".jpg"+"\t"+Cla1+"\n")
		f4.write(path+"/"+sam+"/"+soft+"/Pileup_Image/"+str(i)+".jpg"+"\t"+Cla2+"\n")
		i+=1
		
	bed_f.close()
	return CNV_bed


if __name__ == "__main__":
	bed=sys.argv[1]
	path=sys.argv[2]
	sam=sys.argv[3]
	soft=sys.argv[4]
	Read_bed(bed,path,sam,soft)
