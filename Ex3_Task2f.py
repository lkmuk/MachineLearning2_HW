'''
Ex3::Q2f 
an interpretation to Beta distribution when alpha,beta are positive integer
using Monte-Carlo method.

An example (with plot) is provided at the bottom of this script 
(run only if this script is invoked directly from a Python session)

The Random Variable in question is denoted as $mu_infty$ (defined in the next paragraph)
	The task is to determine its distribution (approximately)
	(which is indeed a Beta as agreed by the MC results, see `simDistr`)
	For each MC trial, 
	a "sufficiently large" number of steps or draws $n$ 
	of a sequentially dependent experiment is performed (explained in the next paragraph).

Experiment description (the implementations are `mysteriousProcess` or `mysteriousProcess2`)
	given an initial positive integer a and b
	which represents the initial number of "hit" outcome(s) and "miss" outcome(s)
	the initial portion of "hit" outcomes in the sample space of each draw 
	is denoted as $mu_0 := a/(a+b)$

	For the next draw, (here $i=1$), 
	the outcome of this draw is drawn uniformly randomly from the previous sample space.
	If the outcome in this draw is a "hit", a := a + 1,
	otherwise, b := b + 1.
	(Conceptually,) the hit ratio of the draw after this one (here $i=2$) would then 
	be updated as $mu_1 := a/(a+b)$

	repeat until $n$ draws are completed

	$mu_n$ will be then be used to approximate (a realization of) $mu_infty$
'''
import numpy as np
import matplotlib.pyplot as plt

def simDistr(a, b,num_trials=None, nums_draws=None):
	"""
	simulate the distribution of the quasi-asymptotic hit ratio mu_infty using MC
	for various values of n (num of steps) in order to compare asmptotic convergence.

	(no sanity check for data type and range...)
	"""
	##############################
	#    Processing Inputs
	#############################
	print('=============================')
	print('Simulating the distribution for...')
	print(' initial hit ratio a/(a+b):\t\t{:.3f}'.format(a/(a+b)))

	if num_trials is None:
		num_trials = 10000 # (should be "sufficient")

	if nums_draws is None:
		nums_draws = [128, 256]

	#############################
	#  var initialization
	############################
	hitRatios_final = np.nan*np.ones([len(nums_draws),num_trials]) #initialization
	num_one_tenth_trials = num_trials/10 # only for progress update

	#############################
	#  MC Simulation for the given a and b
	############################
	# note: the following nested loops can in principle be parallelized
	for num_draws_activeID, num_draws_active in enumerate(nums_draws): # convergency test
		print(' ----------------------------------')
		print(' Now consider ', num_draws_active, ' steps:')
		for trialID in range(num_trials):
			hitRatios_final[num_draws_activeID,trialID] = mysteriousProcess(a, b, num_draws_active)
			
			# progress update per 10% completion increment
			if trialID%num_one_tenth_trials==0: 
				print('  completed {} MC trial(s) out of {} ({:2.0f}%)'.format(\
					trialID+1, num_trials, (trialID+1)/num_trials*100)\
				)
		print(' empirical estimate of long-run hit ratio for {} steps:\t{:.3f}'.format(\
			num_draws_active, \
			np.mean(hitRatios_final[num_draws_activeID,:]))\
		)
	return hitRatios_final


def mysteriousProcess(a,b, num_draws):
    for drawID in range(num_draws):
        if np.random.randint(low=0,high=a+b) >= a: 
       	# predicate = "miss, i.e. getting a white ball in this draw"
        # 	not to mess up with the predicate above!!! 
        #   alternatively, 
        # if np.random.randint(low=1,high=a+b+1) > a:
            b += 1 
        else:
            a += 1
    return a/(a+b)

def mysteriousProcess2(a,b, num_draws):
	"""
	an inefficient implementation (require division per draw)
	"""
	prob_a = a / (a+b)
	for drawID in range(num_draws):
		if np.random.choice([False,True], p=[prob_a,1-prob_a]):
			b += 1
		else:
			a += 1
		prob_a = a / (a+b)
	return prob_a
    
if __name__ == '__main__':
	#############################
	#  change me
	#############################
	ab_to_test=[\
		[3,2], #format: [a,b]
		[2,3],
		[1,1]
	]
	num_trials = 10000
	nums_draws = [100,400,800]

	#############################
	#  simulate for each (a,b) (here, one-by-one)
	#############################
	muN_container = np.nan * np.ones((\
		len(ab_to_test),
		len(nums_draws),
		num_trials
		))
	for ab_set_ID, ab_set in enumerate(ab_to_test):
		muN_container[ab_set_ID,:,:] = \
			simDistr(*ab_set, num_trials=num_trials, nums_draws=nums_draws)

	#############################
	# plot the histograms for mu_infty(a,b)
	#############################
	# each set of (a,b) has a subplot,
	# each suplot contains the histogram of mu_N for the number(s) of draws specified
	# (i.e. meant to show long-run convergence behavior)
	#
	# different sets take different rows
	numBins = 20
	numABsets = len(ab_to_test)
	maxOccurence = int(num_trials/numBins*2)
	fig, axes = plt.subplots(numABsets,1,figsize=(6,4*numABsets),sharex='all', sharey='all')
	for ab_set_ID in range(numABsets):
		axis = axes[ab_set_ID]
		axis.grid(color='gray', linestyle='-', linewidth=0.5)
		for num_draws_activeID in range(len(nums_draws)):
			axis.hist(muN_container[ab_set_ID,num_draws_activeID,:], \
				label="n = {}".format(nums_draws[num_draws_activeID]),
				alpha=0.7,
				bins=numBins
			)
		axis.legend()
		axis.set_ylabel('occurrence')
		axis.set_title("MC results for (a,b) = ({:.0f},{:.0f})".format(*ab_to_test[ab_set_ID]))
		plt.yticks(range(0,maxOccurence+1,int(maxOccurence/4)))
	axes[-1].set_xlabel('realized $\mu_N$')
	plt.subplots_adjust(hspace=1.2)
	plt.tight_layout()
	plt.savefig('Ex3_Task2f_plot.pdf')