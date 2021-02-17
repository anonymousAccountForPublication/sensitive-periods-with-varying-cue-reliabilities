# waht we need: policy and state transition matrix
# combine those two into one ditcionary and read it out into a textfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from PlotCircles import circles
import multiprocessing
import itertools
import os
import random
from operator import itemgetter
from math import *
# set the current working directory

def chunks(l, n):
    if n == 0:
        yield l
    else:
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            if isinstance(l, list):
                yield l[i:i+n]
            else:

                yield l.loc[i:i+n-1].reset_index(drop = True)

def convertValues(valueArr, old_max ,old_min,new_max, new_min):
    minArr =old_min
    maxArr = old_max
    rangeArr = maxArr-minArr
    newRangeArr = new_max-new_min
    result = [((val - minArr)/float(rangeArr))*newRangeArr+new_min for val in valueArr]
    return result



def area_calc(probs, r):
    result = [np.sqrt(float(p))*r for p in probs]
    return result

def duplicates(n):
    counter=Counter(n) #{'1': 3, '3': 3, '2': 3}
    dups=[i for i in counter if counter[i]!=1] #['1','3','2']
    result={}
    for item in dups:
        result[item]=[i for i,j in enumerate(n) if j==item]
    return result

# hepler function for plotting the lines
def isReachable(currentIdent, nextIdentList):
    condition_a = currentIdent*2
    condition_b = condition_a+1
    yVals = [idx for idx,item in enumerate(nextIdentList) if (condition_a in item or condition_b in item)]
    yVals = list(set(yVals))
    return yVals


def joinIndidividualResultFiles(argument, tValues, dataPath):
    # need to provide the dataPath accordingly
    if argument == 'raw':

        resultsDFAll =[]
        for t in tValues:
            print 'Currently aggregating data for time step %s' % t
            # batch level
            resultDFList = [batchPstar for batchPstar in os.listdir(os.path.join(dataPath, '%s' % t))]
            resultDFListSorted = [batchPstar for batchPstar in
                                  sorted(resultDFList, key=lambda x: int(x.replace('.csv', '')))]

            # read and concatenate all csv file for one time step
            resultsDF = pd.concat(
                [pd.read_csv(os.path.join(dataPath, os.path.join('%s' % t, f)),index_col=0).reset_index(drop = True) for f in resultDFListSorted]).reset_index(drop = True)
            resultsDFAll.append(resultsDF)
        finalData = pd.concat(resultsDFAll).reset_index(drop = True)
        finalData.to_csv('finalRaw.csv')

    elif argument == "aggregated":
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalAggregated.csv')
    elif argument == 'plotting':
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalPlotting.csv')

    else:
        print "Wrong argument"

# function for parallelization
def plotLinesCopy(subDF1Identifiers, nextIdentList):
    yvalsIDXAll = []
    if subDF1Identifiers:
        for identSubDF1 in subDF1Identifiers:  # as many lines as unique cue validities
            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
            subList.sort()
            subList2 = list(subList for subList, _ in itertools.groupby(subList))
            yvalsIDXAll.append(subList2)
            del subList
            del subList2
    return yvalsIDXAll

def plotLines(identSubDF1, nextIdentList):
    subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
    subList.sort()
    subList2 = list(subList for subList, _ in itertools.groupby(subList))
    del subList
    return subList2

def func_star(allArgs):
    return plotLines(*allArgs)


def cleanIdentifiers(oldIdentifiers):
    newIdent = [str(ident).replace('[', '').replace(']', '').split(',') for ident in oldIdentifiers]
    newIdent2 = [[int(str(a).replace('.0', '')) for a in subList] for subList in newIdent]

    return newIdent2


def policyPlotReduced(T,r,pE0Arr, pC0E0Arr, tValues, dataPath, lines, argumentR, argumentP, minProb,mainPath, plottingPath):
    # preparing the subplot
    fig, axes = plt.subplots(len(pC0E0Arr), len(pE0Arr), sharex= True, sharey= True)
    fig.set_size_inches(16, 16)
    fig.set_facecolor("white")
    ax_list = fig.axes

    # looping over the paramter space
    iX = 0
    for cueVal in pC0E0Arr: # for each cue validity
        jX = 0
        for pE0 in pE0Arr: # for each prior
            # set the working directory for the current parameter combination
            os.chdir(os.path.join(mainPath,"runTest_%s%s_%s_%s" % (argumentR[0], argumentP[0], pE0, cueVal)))

            ax = ax_list[iX*len(pE0Arr)+jX]

            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            for t in tValues:
                # here is where the relevant files are loaded
                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' %t))
                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, T - 1, 1)
                aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.stateProb >minProb] # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] ==t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t,key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])


                color_palette = {0:'#be0119', 1:'#448ee4', 2:'#000000', 3: '#98568d', 4: '#548d44', -1: '#d8dcd6'}
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])
                area = area_calc(aggregatedResultsDF['stateProb'], r)

                # now plot the developmental trajectories
                circles(aggregatedResultsDF['time'],aggregatedResultsDF['newpE1'], s =area, ax = ax,c = colors, zorder = 2, lw = 0.5)

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.clock()
                for t in np.arange(0,T-1,1):
                    print "Current time step: %s" % t
                    tNext = t+1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t+1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, T - 1, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)
                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time ==1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop = True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' %tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, T - 1, 1)
                        subDF2.reset_index(drop=True, inplace= True)

                        subDF2.drop(index = indices, inplace= True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    subDF1['Identifier'] = cleanIdentifiers(subDF1.Identifier)
                    subDF2['Identifier'] = cleanIdentifiers(subDF2.Identifier)
                    nextIdentList = subDF2['Identifier']

                    yvalsIDXAll = []
                    if t <= 11: # otherwise the overhead for multiprocessing is slowing down the computation
                        for identSubDF1 in list(subDF1.Identifier):
                            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
                            subList.sort()
                            subList2 = list(subList for subList, _ in itertools.groupby(subList))
                            yvalsIDXAll.append(subList2)
                            del subList
                            del subList2
                    else:

                        for identSubDF1 in list(subDF1.Identifier):  # as many lines as unique cue validities
                            pool = multiprocessing.Pool(processes=32)
                            results = pool.map(func_star, itertools.izip(chunks(identSubDF1, 1000), itertools.repeat(nextIdentList)))
                            pool.close()
                            pool.join()
                            resultsUnchained = [item for sublist in results for item in sublist]
                            yvalsIDXAll.append(resultsUnchained)
                            del results
                            del resultsUnchained

                        # process the results
                    yArr = []
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                itertools.chain.from_iterable(yvalsIDXAll[subIDX])]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#e6daa6', zorder=1, lw=0.3) for yArrr in
                         yArr]
                    del yArr
                elapsedTime = time.clock()-startTime
                print "Elapsed time plotting the lines: " + str(elapsedTime)
            #
            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly
            xTuple = [current[0] for current in coordinates]
            yTuple = [current[1] for current in coordinates]
            radii = []
            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i)/sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR= np.sqrt(sum(stateProbPies[idx]))*r
                radii.append(currentR)
                pp,tt = ax.pie(pieFracs,colors = colorsPies, radius = currentR ,center = coordinates[idx], wedgeprops= {'linewidth':0.0, "edgecolor":"k"})
                [p.set_zorder(3+len(coordinates)-idx) for p in pp]


            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.sca(ax)
            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (T)/float(2)
            yLabels = convertValues([1,midPoint,T-1], T-1,1,1,0)
            # removing frame around the plot
            plt.ylim(0.4, T-1+0.5)
            plt.xlim(-0.6,T-1+0.5)

            if iX == 0:
                plt.title(str(1-pE0), fontsize = 20)


            if iX == len(pC0E0Arr) - 1:
                plt.xticks([], fontsize = 15)


            else:
                ax.get_xaxis().set_visible(False)


            if jX == 0 and iX == len(pC0E0Arr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.yticks([1, midPoint, T - 1], yLabels, fontsize=12)
                plt.ylabel('estimate', fontsize=20, labelpad=10)


            if jX == len(pE0Arr) -1:
                plt.ylabel(str(cueVal),  fontsize = 20,labelpad = 15, rotation = 'vertical')
                ax.yaxis.set_label_position("right")

            ax.set_aspect('equal')

            jX += 1
        iX += 1

    plt.suptitle('prior probability', fontsize = 20)
    fig.text(0.98,0.5,'cue reliability', fontsize = 20, horizontalalignment = 'right', verticalalignment = 'center', transform=ax.transAxes, rotation = 'vertical')
    resultPath = os.path.join(mainPath, plottingPath)
    plt.savefig(os.path.join(resultPath,'DevelopmentalTrajectoryReduced.png'), dpi = 1200)







