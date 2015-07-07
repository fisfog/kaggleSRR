import numpy as np
import pandas as pd
import math


def ensemble(predlist,outfile,w):
	pred_df = []
	for f in predlist:
		pred_df.append(pd.read_csv(f))

	all_df = pred_df[0]
	for i in xrange(1,len(pred_df)):
		all_df = pd.merge(all_df,pred_df[i],on='id')
	data = all_df.drop(['id'],axis=1).values
	idx = all_df.id.values
	print idx
	pred = []
	for i in xrange(len(all_df)):
		# pred.append(int(math.floor(data[i].mean())))
		p3 = 0
		for j in xrange(len(predlist)):
			p3 += w[j]*data[i][j]
		pred.append(int(np.floor(p3/sum(w))))
	submission = pd.DataFrame({"id": idx, "prediction": pred})
	submission.to_csv(outfile, index=False)
