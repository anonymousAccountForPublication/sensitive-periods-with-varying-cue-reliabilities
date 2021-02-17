'''
Add some additional publication plots
    specifically a plot showing in the left panel the var-cue reliability function
    and in the right panel the cummulative var cue reliability function
'''
import cPickle as pickle
import matplotlib.pyplot as plt
import os
import numpy as np




def calcBeliefTrajectory(pE1, pC1E1Arr, arg):
    results = []

    for pC1E1 in pC1E1Arr:
        for idx in range(arg):
            b0, b1 = BayesianUpdating(1 - pE1, pE1, 1 - pC1E1, pC1E1)
            pE1 = b1
        results.append(b1)

    return results




def BayesianUpdating(pE0, pE1, pDE0, pDE1):
    # pE0 is the evolutionary prior vor environment 1
    # pE1 is the evolutionary prior for environment 2
    # pDE0 and pDE1 are the probabilities of obtaining the data given environment 0 or 1 respectively (likelihood)
    p_D = pDE0 * pE0 + pDE1 * pE1
    b0_D = (pDE0 * pE0) / p_D
    b1_D = (pDE1 * pE1) / p_D
    return b0_D, b1_D


penalty = 'linear'
reward = "linear"


prior = 0.5
priorArr = [0.5,0.7,0.9]
cueRelArr = ['triangular', 'increasing','decreasing']

dataPath = '/home/nicole/PhD/Proejct1 varying cue reliabilities/temporaryVarCue/21timesteps/resultsMax075_21T' #/runTest_ll_0.5_decreasing

fig, axes = plt.subplots(2, 3, sharex=True , sharey=False)
fig.set_size_inches(30, 20)
ax_list = fig.axes
ix = 0
colors = np.array([0.2,0.5,0.7])
linestyles = [':','--','-']


axisArr = [0,2]
for val in axisArr:
    ax = ax_list[val]
    plt.sca(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



ax = ax_list[1]
plt.sca(ax)

for cueVal in cueRelArr:
    pC0E0 = pickle.load(open(os.path.join(dataPath,"runTest_ll_%s_%s/pC0E0dict.p" %(1-prior, cueVal)), "rb"))
    ax.plot(pC0E0.keys(),pC0E0.values(), linestyle=linestyles[ix], linewidth=2, color = [colors[ix]]*3)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ix+=1

ax.plot(pC0E0.keys(),[np.mean(pC0E0.values())]*len(pC0E0.keys()), linestyle='-.', linewidth=2, color = [0]*3)

maxCueRel = max(pC0E0.values())
minCueRel = min(pC0E0.values())
ax.plot(pC0E0.keys(),[maxCueRel]*len(pC0E0.keys()), ls = '--', lw = 0.8, color = '#B8B8B8', zorder = -1)
ax.plot(pC0E0.keys(), [minCueRel] * len(pC0E0.keys()), ls='--', lw=0.8, color='#B8B8B8', zorder = -1)




# This is for the first one
ax.set_ylim(0.5,1)
ax.set_xlim(0.9,21.1)
ax.set_xticks(pC0E0.keys())
ax.set_ylabel("cue reliability", fontsize=20, labelpad=10)
ax.set_xlabel("time", fontsize=20)
ax.set_yticks([0.5,minCueRel,maxCueRel,1])
ax.tick_params(labelsize = 15)
ax.set_aspect(35)





for priorIX,prior in enumerate(priorArr):
    ax = ax_list[3+priorIX]
    plt.sca(ax)
    ix = 0
    for cueVal in cueRelArr:
        pC0E0 = pickle.load(open(os.path.join(dataPath, "runTest_ll_%s_%s/pC0E0dict.p" % (0.5, cueVal)), "rb"))

        ax.plot(pC0E0.keys(), calcBeliefTrajectory(prior, pC0E0.values(),1), linestyle=linestyles[ix], linewidth=2, color = [colors[ix]]*3,label = cueVal)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        if priorIX == 0:
            ax.spines['left'].set_visible(True)
        else:
            ax.spines['left'].set_visible(False)
        ix+=1

    ax.plot(pC0E0.keys(),calcBeliefTrajectory(prior,[np.mean(pC0E0.values())]*len(pC0E0.keys()),1), linestyle='-.', linewidth=2, color = [0]*3, label = 'average cue reliability')



for priorIX,prior in enumerate(priorArr):
    ax = ax_list[3+priorIX]
    plt.sca(ax)

    ax.set_ylim(0.1,1.05)
    ax.set_xlim(0.9,21.1)
    ax.set_xticks(pC0E0.keys())
    ax.tick_params(labelsize = 15)
    ax.set_title("prior: %s" %str(prior), fontsize=20)
    ax.set_aspect(20)

    if priorIX == 0:
        ax.legend(loc = 4, fontsize = 20)
        ax.set_xlabel("time", fontsize = 20)
        ax.set_ylabel("posterior", fontsize=20, labelpad=10)

plt.savefig(os.path.join(dataPath, 'cueValPlot%s.png' %priorArr), dpi = 900)
plt.close()

exit()
"""
plot how multiple cues of different reliabilities 
"""

penalty = 'linear'
reward = "linear"

prior = 0.5

fig, axes = plt.subplots(1, 1)
fig.set_size_inches(20, 20)
colors = np.array([0.2,0.5,0.7])
linestyles = ['-','--','-.']
cueReliabilities = [0.55,0.75,0.95]
T = 20
tValues = np.arange(1,T+1,1)
# first plot the three cue reliabilities for comparison purposes
ix = 0
for cueRel in cueReliabilities:
    plt.plot(tValues, calcBeliefTrajectory(prior,[cueRel]*T,1), linewidth = 3, color = [0]*3, linestyle = linestyles[ix], label = "Cue reliability: %s with 1 cue"%(cueRel))
    ix+=1

ix = 0
for idx in np.arange(2,5,1):
    plt.plot(tValues, calcBeliefTrajectory(prior, [cueReliabilities[0]] * T, idx), linewidth=3, color=[colors[ix]] * 3,
             label="Cue reliability: %s with %s cues" % (cueReliabilities[0], idx), linestyle = linestyles[0], marker = '.',markersize =15)
    ix += 1
plt.legend(loc = 4, fontsize = 20)

axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_visible(True)
axes.spines['left'].set_visible(False)
plt.ylim(0.5,1.05)
plt.xlim(0.9,20.1)
plt.xticks(np.arange(1,21,1), fontsize = 14)
plt.yticks(np.arange(0.5,1.01,0.1), fontsize = 14)
plt.xlabel("Time", fontsize = 20, labelpad = 20)
plt.ylabel("Posterior", fontsize = 20,labelpad = 20)
plt.savefig(os.path.join(dataPath, 'frequentCues.pdf'), dpi = 1200)
plt.close()





