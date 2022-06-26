import pandas as pd

sub1 = pd.read_csv('section2_v_f05.txt',sep='\t')
sub2 = pd.read_csv('section2_lh.txt',sep='\t')
sub1['Id'] = sub1['Id'].astype(int)
sub = sub2.merge(sub1,on='Id')
sub['result'] = sub['result_x']*0.6 + sub['result_y']*0.4
sub[['Id','result']].to_csv('section2.txt',sep='\t',index=False)