#!/usr/bin/python3

#############
#  options  #
#############

CDR3='cdr3-bs-ff'

###############
#  variables  #
###############

CC_VEC = ["DP_pre", "DP_DBN", "DP_pos", "CD4_spl", "CD4_negS", "CD8_spl", "CD8_negS" ]
HEADERS = [ "DP pre", "DP dbn", "DP pos", "CD4 spl", "CD4 negS", "CD8 spl", "CD8 negS" ]

aux = ['#0088ff', "#a5c630","#488f31","#00d3d6","#ef5675","#bc5090","#ffa600","#ff764a"]
EXP_KOL = dict(zip(['gen']+HEADERS, aux))
EXP_KOL[r'$P_{gen}$'] = EXP_KOL['gen']
EXP_KOL[r'$P_{\rm gen}$'] = EXP_KOL['gen']

###############
#  practical  #
###############
    
CC_to_HEAD = dict(zip(CC_VEC,HEADERS))
CC_to_HEAD['gen']=r'$P_{\rm gen}$'
HEAD_to_CC = dict(zip(HEADERS,CC_VEC))
HEAD_to_CC['gen']='gen'

#####################
#  graph standards  #
#####################

GREYS_DIC = {'signal': '#282828', 'pidgeon': '#606e8c', 'silver': '#d4d4d4' }