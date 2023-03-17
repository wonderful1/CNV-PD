#!/usr/bin/env python3

#system module
import pysam as ps
import pandas as pd
import argparse
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

#user-defined module
import Read_File as rf
import Get_Base_Rgb as gbr
import Get_Feature as gf


#Defind the USAGE
class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
		def add_usage(self, usage, actions, groups, prefix=None):
				if prefix is None:
						prefix = 'Usage: '
						return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)

ap = argparse.ArgumentParser(formatter_class=CapitalisedHelpFormatter)


ap.add_argument(
		'BamFileName',
		metavar='BamFileName',
		help='Bamfile to be used here'
)

ap.add_argument(
		'-b',
		'--bed',
		type=str,
		default=None,
		metavar='bed',
		help='path to a .bed file with locations of CNVs. The second column should be CNV start, and the third column should be CNV end '
)

ap.add_argument(
		'-bas',
		'--bas_file',
		type=str,
		default=None,
		metavar='bas_file',
		help='path to a .bas file'
)



ap.add_argument(
		'-bs',
		'--bin-size',
		type=int,
		default=100,
		metavar='bin_size',
		help='the bin size that each bin will represent, in bases'
)

ap.add_argument(
		'-k',
		'--step-size',
		type=int,
		default=50,
		metavar='step_size',
		help='the step-size for taking sliding windows to create bins'
)


ap.add_argument(
		'-o',
		'--output',
		type=str,
		default='.',
		metavar='outdir',
		help='the directory into which output files will go'
)

args = ap.parse_args()

#some module
# init image
def init_pic(row,col,th,fig, flag):
	if flag=='2d':
		ax = fig.add_subplot(row, col, th)
		ax.get_xaxis().get_major_formatter().set_useOffset(False)
		return ax
	elif flag == '3d':
		ax = fig.add_subplot(row, col, th, projection='3d')
		ax.get_xaxis().get_major_formatter().set_useOffset(False)
		return ax


def draw_bar(clip_dict_record, pile_record,del_pos_np_start, deletion_length,outdir,i):
	blank = Image.new("RGB",[1000, 500],"white")
	wid_x=1000/int(deletion_length)
	wid_y=5

	pile_record_len = len(pile_record)
	drawObject = ImageDraw.Draw(blank)
	y_start_index = 0
	old_x_start = wid_x
	for j in range(pile_record_len):
			x_start = (pile_record[j][0] - del_pos_np_start)*wid_x + wid_x
			if old_x_start == x_start:
				old_x_start = x_start
				y_start = wid_y + y_start_index*wid_y
				y_start_index += 1
				x_end = x_start + wid_x
				y_end = y_start + wid_y
				if pile_record[j][0] in clip_dict_record:
					base_rgb = gbr.Get_Rgb(-clip_dict_record[pile_record[j][0]],pile_record[j])
				else:
					base_rgb = gbr.Get_Rgb(0,pile_record[j])
				drawObject.rectangle((x_start,y_start, x_end, y_end),fill=base_rgb)
			elif old_x_start != x_start:
				old_x_start = x_start
				y_start_index = 0
				y_start = wid_y + y_start_index*wid_y
				y_start_index += 1
				x_end = x_start + wid_x
				y_end = y_start + wid_y

				if pile_record[j][0] in clip_dict_record:
					base_rgb = gbr.Get_Rgb(-clip_dict_record[pile_record[j][0]],pile_record[j])
				else:
					base_rgb = gbr.Get_Rgb(0,pile_record[j])
				drawObject.rectangle((x_start,y_start, x_end, y_end),fill=base_rgb)

	if os.path.exists(outdir+'/Pileup_Image/'):
		blank.save(outdir+'/Pileup_Image/' + str(i) + ".jpg")
	else:
		os.mkdir(outdir + '/Pileup_Image/')
		blank.save(outdir + "/Pileup_Image/" + str(i) + ".jpg")



def get_image(vcf_del,sam_file,RG_bas_dict,outdir):   
	vcf_len = len(vcf_del)
	print("vcf_len %d"%vcf_len)
	chr_id = vcf_del[0][0]
	del_pos = []
	for i in range(vcf_len):
		print("i = %d"%i)
		read_depth = gf.Get_Depth(sam_file,vcf_del[i][0],int(vcf_del[i][1])-500,int(vcf_del[i][2])+500) #get_depth(sam,del_chr,start-200,end +200)
		seq_depth = []

		len_deletion = len(read_depth)
		for j in range(len_deletion):
			seq_depth.append((int(vcf_del[i][1])-500+j, int(read_depth[j])))

		pile_feature,pair_record,qua_record,pe_record,clip_record = gf.Get_Feature(sam_file,vcf_del[i][0],int(vcf_del[i][1])-500,int(vcf_del[i][2])+500,RG_bas_dict) 

	####1. draw read depth and split read plot
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
		fig = plt.figure()
		ax = init_pic(6,1,1,fig,'2d')
		seq_depth_array = np.array(seq_depth)
		ax.plot(seq_depth_array[:,0],seq_depth_array[:,1],color='r') 
		ax = init_pic(6,1,2,fig,'2d')
		seq_clip_array = np.array(clip_record)
		ax.plot(seq_clip_array[:,0],seq_clip_array[:,1],color='g')
		ax = init_pic(6,1,3,fig,'2d')
		seq_pair_array = np.array(pair_record)
		ax.plot(seq_pair_array[:,0],seq_pair_array[:,1],color='b')
		ax = init_pic(6,1,4,fig,'2d')
		seq_qua_array = np.array(qua_record)
		ax.plot(seq_qua_array[:,0],seq_qua_array[:,1],color='black')
		ax = init_pic(6,1,5,fig,'2d')
		seq_pe_array = np.array(pe_record)
		ax.plot(seq_pe_array[:,0],seq_pe_array[:,1],color='grey')

		ax0 = init_pic(6,1,6,fig,'2d')
		ax0.bar(seq_depth_array[:500,0],seq_depth_array[:500,1],color='b')
		ax0.bar(seq_depth_array[500:-500,0],seq_depth_array[500:-500,1],color='r')
		ax0.bar(seq_depth_array[-500:,0],seq_depth_array[-500:,1],color='b')
		
		if os.path.exists(outdir+'/RD_SR/'):  
			fig.savefig(outdir+"/RD_SR/" + str(i) + ".jpg")
		else:
			os.mkdir(outdir+'/RD_SR/')
			fig.savefig(outdir+"/RD_SR/" + str(i) + ".jpg")
		
	####2. draw pileup image
#		print(chr_id)
		CNV_length = int(vcf_del[i][2])-int(vcf_del[i][1])+1000
		clip_dict = dict(clip_record)
		draw_bar(clip_dict, pile_feature,int(vcf_del[i][1])-500 , CNV_length,outdir,i)
		



if __name__ == "__main__":
	#reading file
	CNV_bed=rf.Read_bed(args.bed)
	bamfile=rf.Read_bam(args.BamFileName)
	RG_bas=rf.Read_bas(args.bas_file)
	outdir=args.output
	#drawing Image
	get_image(CNV_bed,bamfile,RG_bas,outdir)

