import numpy as np
import pandas as pd

def Get_Depth(bamfile,chrom,pos_l,pos_r):
	depth = bamfile.count_coverage(chrom,pos_l,pos_r)  
	sum_depth = np.array(list(depth)).sum(axis=0).tolist()
	return sum_depth
#Get_Depth(bamfile,"19",60006,60100)

def get_sum(temp):
	record_np = np.array(temp)
	df = pd.DataFrame(record_np)
	record_df = df.groupby(0).sum()
	temp = record_df.reset_index()
	record = np.array(temp).tolist()
	return record


def Get_Feature(bamfile,chrom,pos_l,pos_r,RG_bas_dict):
	'''1)proper_pair  2)mapping_quality  3)insert size  4)split read '''
	pile_feature=[]
	clip_temp=[]
	pair_temp=[]
	pe_temp=[]
	qua_temp=[]
	for pileupcolumn in bamfile.pileup(chrom, pos_l, pos_r):
		if(pileupcolumn.pos>=pos_l and pileupcolumn.pos <= pos_r):
			for pileupread in pileupcolumn.pileups:
				if pileupread.alignment.cigarstring != None and  pileupread.query_position != None :
					split_read=(1 if(pileupread.alignment.get_cigar_stats()[0][4]>= 2) else 0)
					
					
					mean,stdev=RG_bas_dict[pileupread.alignment.get_tag("RG")].split("\t")
					mean=float(mean)
					stdev=float(stdev)
					ab_ins_size=(1 if ((abs(pileupread.alignment.next_reference_start-pileupread.alignment.get_reference_positions()[1]+pileupread.alignment.query_alignment_length)-mean) >= 3*stdev) else 0)
					   
					pile_result = (pileupcolumn.pos,pileupread.alignment.is_proper_pair,pileupread.alignment.mapping_quality,ab_ins_size,split_read,pileupread.alignment.query_sequence[pileupread.query_position])
					pile_feature.append(pile_result)
					
					pair_is=(0 if(pileupread.alignment.is_proper_pair) else 1)
					pe_is=ab_ins_size
					qua_is=(0 if(pileupread.alignment.mapping_quality> 0) else 1)
					clip_temp.append((pileupcolumn.pos,split_read))
					qua_temp.append((pileupcolumn.pos,qua_is))
					pe_temp.append((pileupcolumn.pos,pe_is))
					pair_temp.append((pileupcolumn.pos,pair_is))
	#paired
	pair_record = get_sum(pair_temp)
	#quality
	qua_record=get_sum(qua_temp)
	#pair end
	pe_record=get_sum(pe_temp)
	#split read
	clip_record = get_sum(clip_temp)

	print("pilelen is %d"%len(pile_feature))
	return pile_feature,pair_record,qua_record,pe_record,clip_record
#Get_Feature(bamfile,"19",60006,60100,RG_bas)
