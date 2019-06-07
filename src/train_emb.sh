# env python3.6

# step 1 : get alldata.txt for training doc2vecC and ID list for saving news vector to dict later
# input meta_data, stops.txt, dict.txt.big
python get_trainTXT_IDList.py -i $1
wait
# output alldata.txt, ID_list_off.pkl

# step 2 : train doc2vecC to get news vector 
# input: alldata.txt, doc2vecc.c                                                                                                                                         
bash go.sh 
wait
# output: docvectors.txt, wordvectors.txt

# step 3 : save news vector to dictionary
# input: ID_list.pkl, docvectors.txt
python doc2vecC_to_dict.py
# output: ID_list.pkl, news_vec_pool.txt