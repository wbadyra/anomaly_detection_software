Fault detection software, based on clustering, neural networks and other machine learning methods.

For now results and experiments are a little scattered and unordered. However in each experiment are
results for all possible combinations of parameters. For example in PCA folder (in every experiment)
there are two files: results_PCA.csv, which contains records with PCA method selected, and 
stats_results_PCA.csv, in which are average statistics of it. There are also directories, ex. DBscan.
In it there are all results which contains PCA as reduction method and DBscan as model used and etc.
In experiment dir name is also stated, if mistakes were inserted on same samples (same_time), or not
(diff_time). Each experiments inserted faults params:

EXP 1, 2, 3:
	errors intensity -> 2.0
	occuring percentage -> 5%
	
EXP 4:
	errors intensity -> 1.5
	occuring percentage -> 5%

EXP 5:
	errors intensity -> 3
	occuring percentage -> 5%

