import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import cPickle as pickle
import ternary
import seaborn as sns
from cmocean import cm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

"""""
This script will run experimental twin studies to test how sensitive different optimal policies are to cues
    in these experiments I simulate agents in specific environments based on the previously calculated optimal policies
    importantly I will try to use the finalRAW file rather than the dictionary files 
    
    procedure:
    - simulate twins who are identical up to time period t 
    - keep one twin ("original") in its natal patch
    - send the other twin ("doppelgaenger") to mirror patch 
    - doppelgaenger receives opposite (yoked) cues from the original twin
        the cues are opposite but not from the opposite patch
"""""


def setGlobalPolicy(policyPath):
    global policy
    policy = pd.read_csv(policyPath, index_col=0).reset_index(drop=True)


def func_star(allArgs):
    return simulateTwins(*allArgs)


def func_star2(allArgs):
    return simulateExperimentalTwins(*allArgs)


def updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker):
    optDecisions = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                      subDF['yw'] == phenotypeTracker[idx, 2])
                              & (subDF['Identifier'] == identTracker[idx])]['cStar'].item() for idx in
                    simValues]

    # additionally keep track of the posterior belief
    posBelief = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                   subDF['yw'] == phenotypeTracker[idx, 2])
                           & (subDF['Identifier'] == identTracker[idx])]['pE1'].item() for idx in
                 simValues]
    # post process optimal decisions
    optDecisionsNum = [
        int(a) if not '(' in str(a) else int(np.random.choice(str(a).replace("(", "").replace(")", "").split(","))) for
        a in
        optDecisions]
    # update phenotype tracker
    idx0 = [idx for idx, val in enumerate(optDecisionsNum) if val == 0]
    if idx0:
        phenotypeTracker[idx0, 0] += 1

    idx1 = [idx for idx, val in enumerate(optDecisionsNum) if val == 1]
    if idx1:
        phenotypeTracker[idx1, 1] += 1

    idx2 = [idx for idx, val in enumerate(optDecisionsNum) if val == 2]
    if idx2:
        phenotypeTracker[idx2, 2] += 1

    return phenotypeTracker, posBelief


def updateIdentiyTracker(identTracker, cues):
    newIdentTracker = [2 * i if c == 0 else (2 * i + 1) for i, c in zip(identTracker, cues)]

    return newIdentTracker


def simulateExperimentalTwins(tAdopt, twinNum, env, cueReliability, lag, T, adoptionType, endOfExposure):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability
    :return: phenotypic distance between pairs of twins
    """
    T = T +lag -1
    tValues = np.arange(1, tAdopt, 1)
    if env == 1:
        pC1Start = cueReliability[1]  # take the very first cue reliability
    else:
        pC1Start = 1 - cueReliability[1]
    pC0Start = 1 - pC1Start

    cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

    # need to reverse the last update
    if adoptionType == "yoked":
        oppositeCues = 1 - cues

    elif adoptionType == "oppositePatch":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
        oppositeCues = np.array(oppositeCues)
    elif adoptionType == "deprivation":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
        oppositeCues = np.array(oppositeCues)

    else:
        print "wrong input argument to adoption type!"
        exit(1)

    cueTracker = np.zeros((twinNum, 2))
    cueTracker[:, 0] = 1 - cues
    cueTracker[:, 1] = cues
    phenotypeTracker = np.zeros((twinNum, 3))

    identTracker = cues
    if len(tValues) != 0:
        identTrackerDoppel = cues
    else:
        identTrackerDoppel = oppositeCues
    simValues = np.arange(0, twinNum, 1)

    for t in tValues:
        # now we have to recompute this for every timestep
        if env == 1:
            pC1Start = cueReliability[t]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[t]

        pC0Start = 1 - pC1Start

        np.random.seed()
        # print "currently simulating time step: " + str(t)
        subDF = policy[policy['time'] == t].reset_index(drop=True)
        # next generate 10000 new cues
        # generate 10000 optimal decisions

        # probably need an identity tracker for the new policies
        cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

        cues = np.array(cues)
        phenotypeTracker, __ = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker)
        # update identity tracker for new cues
        identTracker = updateIdentiyTracker(identTracker, cues)
        if t != tValues[-1]:
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cues)
        else:
            # last step is where we get yoked opposite cues
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
        # update cue tracker
        cueTracker[:, 0] += (1 - cues)
        cueTracker[:, 1] += cues

    # post adoption period
    # continue here
    originalTwin = np.copy(phenotypeTracker)
    doppelgaenger = np.copy(phenotypeTracker)

    restPeriod = np.arange(tAdopt, tAdopt + lag, 1)

    # setting up the matrix for the yoked opposite cues
    cueTrackerDoppel = np.copy(cueTracker)

    cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
    cueTrackerDoppel[:, 1] += -cues + oppositeCues

    for t2 in restPeriod:
        if env == 1:
            pC1Start = cueReliability[t2]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[t2]

        pC0Start = 1 - pC1Start

        np.random.seed()
        subDF = policy[policy['time'] == t2].reset_index(drop=True)
        # probably need an identity tracker for the new policies
        cuesOriginal = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
        cuesOriginal = np.array(cuesOriginal)

        if adoptionType == "yoked":
            oppositeCues = 1 - cuesOriginal
        elif adoptionType == "oppositePatch":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
            oppositeCues = np.array(oppositeCues)
        else:  # adoptionType = deprivation
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)
        # update the phenotypes of the twins
        originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)
        identTracker = updateIdentiyTracker(identTracker, cuesOriginal)

        doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel, identTrackerDoppel)

        if t2 != restPeriod[-1]:
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
        else:
            # last step is where we get yoked opposite cues
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cuesOriginal)

        # update cue tracker
        cueTracker[:, 0] += (1 - cuesOriginal)
        cueTracker[:, 1] += cuesOriginal

        cueTrackerDoppel[:, 0] += (1 - oppositeCues)
        cueTrackerDoppel[:, 1] += oppositeCues

    restPeriodReunited = np.arange(tAdopt + lag, T + 1, 1)
    # need to reverse the last update
    cueTrackerDoppel[:, 0] += -(1 - oppositeCues) + (1 - cuesOriginal)
    cueTrackerDoppel[:, 1] += -(oppositeCues) + cuesOriginal


    if not endOfExposure:  # this means we want to measure phenotypic distance at the end of onotgeny
        for t3 in restPeriodReunited:
            # they will receive the same cues again

            if env == 1:
                pC1Start = cueReliability[t3]  # take the very first cue reliability
            else:
                pC1Start = 1 - cueReliability[t3]

            pC0Start = 1 - pC1Start

            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t3].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            # probably need an identity tracker for the new policies
            cuesOriginal = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
            cuesOriginal = np.array(cuesOriginal)

            originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)
            # update identity tracker for new cues
            identTracker = updateIdentiyTracker(identTracker, cuesOriginal)
            doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel, identTrackerDoppel)
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cuesOriginal)

            # update cue tracker
            cueTracker[:, 0] += (1 - cuesOriginal)
            cueTracker[:, 1] += cuesOriginal

            cueTrackerDoppel[:, 0] += (1 - cuesOriginal)
            cueTrackerDoppel[:, 1] += cuesOriginal

    return originalTwin, doppelgaenger


def simulateTwins(tAdopt, twinNum, env, cueReliability, adopt, T, adoptionType):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability array!
    :return: phenotypic distance between pairs of twins
    """
    if adopt:
        tValues = np.arange(1, tAdopt, 1)
        if env == 1:
            pC1Start = cueReliability[1]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[1]
        pC0Start = 1 - pC1Start

        cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

        # need to reverse the last update
        if adoptionType == "yoked":
            oppositeCues = 1 - cues
        elif adoptionType == "oppositePatch":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
            oppositeCues = np.array(oppositeCues)
        elif adoptionType == "deprivation":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)
        else:
            print "wrong input argument to adoption type!"
            exit(1)

        cueTracker = np.zeros((twinNum, 2))
        cueTracker[:, 0] = 1 - cues
        cueTracker[:, 1] = cues
        phenotypeTracker = np.zeros((twinNum, 3))
        posBeliefTracker = [0] * twinNum

        identTracker = cues
        if len(tValues) != 0:
            identTrackerDoppel = cues
        else:
            identTrackerDoppel = oppositeCues
        simValues = np.arange(0, twinNum, 1)
        for t in tValues:
            # now we have to recompute this for every timestep
            if env == 1:
                pC1Start = cueReliability[t]  # take the very first cue reliability
            else:
                pC1Start = 1 - cueReliability[t]

            pC0Start = 1 - pC1Start

            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            # probably need an identity tracker for the new policies
            cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

            cues = np.array(cues)
            phenotypeTracker, posBeliefTracker = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker,
                                                                 identTracker)
            # update identity tracker for new cues
            identTracker = updateIdentiyTracker(identTracker, cues)
            if t != tValues[-1]:
                identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cues)
            else:
                # last step is where we get yoked opposite cues
                identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
            # update cue tracker
            cueTracker[:, 0] += (1 - cues)
            cueTracker[:, 1] += cues

        # post adoption period
        # continue here
        originalTwin = np.copy(phenotypeTracker)
        doppelgaenger = np.copy(phenotypeTracker)

        posBeliefTrackerOrg = np.zeros((twinNum, T + 1 - tAdopt + 1))
        posBeliefTrackerDG = np.zeros((twinNum, T + 1 - tAdopt + 1))

        # for the first time point where twins are separated the whole time we only add a placeholder for the prior
        # an array of zeros; therefore the postprocessinf needs to be doner atfer the arguments have been returned
        posBeliefTrackerOrg[:, 0] = posBeliefTracker
        posBeliefTrackerDG[:, 0] = posBeliefTracker
        del posBeliefTracker

        restPeriod = np.arange(tAdopt, T + 1, 1)

        # setting up the matrix for the yoked opposite cues
        cueTrackerDoppel = np.copy(cueTracker)

        cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
        cueTrackerDoppel[:, 1] += -cues + oppositeCues

        for t2 in restPeriod:  # this is where adoption starts
            if env == 1:
                pC1Start = cueReliability[t2]  # take the very first cue reliability
            else:
                pC1Start = 1 - cueReliability[t2]

            pC0Start = 1 - pC1Start

            np.random.seed()
            subDF = policy[policy['time'] == t2].reset_index(drop=True)
            # probably need an identity tracker for the new policies
            cuesOriginal = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
            cuesOriginal = np.array(cuesOriginal)

            if adoptionType == "yoked":
                oppositeCues = 1 - cuesOriginal
            elif adoptionType == "oppositePatch":
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
                oppositeCues = np.array(oppositeCues)
            else:  # adoptionType = deprivation
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                oppositeCues = np.array(oppositeCues)
            # update the phenotypes of the twins
            originalTwin, posBeliefOrg = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)
            identTracker = updateIdentiyTracker(identTracker,
                                                cuesOriginal)
            posBeliefTrackerOrg[:, t2 - tAdopt + 1] = posBeliefOrg

            doppelgaenger, posBeliefDG = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel,
                                                         identTrackerDoppel)
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
            posBeliefTrackerDG[:, t2 - tAdopt + 1] = posBeliefDG

            # update cue tracker
            cueTracker[:, 0] += (1 - cuesOriginal)
            cueTracker[:, 1] += cuesOriginal

            cueTrackerDoppel[:, 0] += (1 - oppositeCues)
            cueTrackerDoppel[:, 1] += oppositeCues

            # TODO reduce the amount of data stored for the posterior belief tracking

            # store the very first phenotype following adotption to limit the amount of data you need to store
            if t2 == tAdopt:
                originalTwinTemp = np.copy(originalTwin)
                doppelgaengerTemp = np.copy(doppelgaenger)

        return originalTwin, doppelgaenger, posBeliefTrackerOrg, posBeliefTrackerDG, originalTwinTemp, doppelgaengerTemp


    else:  # to just calculate mature phenotypes and rank order stability

        tValues = np.arange(1, T + 1, 1)
        if env == 1:
            pC1Start = cueReliability[1]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[1]
        pC0Start = 1 - pC1Start
        cuesSTart = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
        cueTracker = np.zeros((twinNum, 2))
        cueTracker[:, 0] = 1 - cuesSTart
        cueTracker[:, 1] = cuesSTart
        phenotypeTracker = np.zeros((twinNum, 3))
        phenotypeTrackerTemporal = np.zeros((twinNum, 3, T))
        posBeliefTrackerTemporal = np.zeros((twinNum, T))

        # introduce identity tracker:
        identTracker = cuesSTart
        simValues = np.arange(0, twinNum, 1)
        for t in tValues:
            # now we have to recompute this for every timestep
            if env == 1:
                pC1Start = cueReliability[t]  # take the very first cue reliability
            else:
                pC1Start = 1 - cueReliability[t]

            pC0Start = 1 - pC1Start

            np.random.seed()
            subDF = policy[policy['time'] == t].reset_index(drop=True)

            cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
            cues = np.array(cues)
            # print identTracker
            phenotypeTracker, posBelief = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker)
            phenotypeTrackerTemporal[:, :, t - 1] = np.copy(phenotypeTracker)
            posBeliefTrackerTemporal[:, t - 1] = np.copy(posBelief)
            identTracker = updateIdentiyTracker(identTracker, cues)

            # update cue tracker
            cueTracker[:, 0] += (1 - cues)
            cueTracker[:, 1] += cues

        # successfully computed mature phenotypes
        return phenotypeTracker, phenotypeTrackerTemporal, posBeliefTrackerTemporal


def runExperimentalTwinStudiesParallel(tAdopt, twinNum, env, pE1, pC1E1, lag, T, resultsPath, argumentR, argumentP,
                                       adoptionType, endOfExposure):
    policyPath = os.path.join(resultsPath, 'runTest_%s%s_%s_%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE1, pC1E1))
    setGlobalPolicy(policyPath)
    simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    # load the cue reliability array
    pC1E1Dict = pickle.load(
        open(os.path.join(resultsPath, 'runTest_%s%s_%s_%s/pC1E1dict.p' % (argumentR[0], argumentP[0], pE1, pC1E1)),
             "rb"))

    pool = Pool(processes=12)

    results = pool.map(func_star2, itertools.izip(itertools.repeat(tAdopt),
                                                  simulationChunk, itertools.repeat(env),
                                                  itertools.repeat(pC1E1Dict), itertools.repeat(lag),
                                                  itertools.repeat(T), itertools.repeat(adoptionType),
                                                  itertools.repeat(endOfExposure)))
    pool.close()
    pool.join()

    results1, results2 = zip(*results)
    return np.concatenate(results1), np.concatenate(results2)


def runTwinStudiesParallel(tAdopt, twinNum, env, pE1, pC1E1, adopt, T, resultsPath, argumentR, argumentP, adoptionType,
                           allENV):
    policyPath = os.path.join(resultsPath, 'runTest_%s%s_%s_%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE1, pC1E1))
    setGlobalPolicy(policyPath)
    simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    # load the cue reliability array
    if not allENV:
        pC1E1Dict = pickle.load(
            open(os.path.join(resultsPath, 'runTest_%s%s_%s_%s/pC1E1dict.p' % (argumentR[0], argumentP[0], pE1, pC1E1)),
                 "rb"))

        pool = Pool(processes=12)

        if adopt:
            results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                         simulationChunk, itertools.repeat(env),
                                                         itertools.repeat(pC1E1Dict), itertools.repeat(adopt),
                                                         itertools.repeat(T), itertools.repeat(adoptionType)))
            pool.close()
            pool.join()

            # results1, results2 refer to the phenotypes of orginals and clones
            # results3, results4 refer to the belief matrices of original and clone; shape: numAgents x separationTime +1
            results1, results2, results3, results4, results5, results6 = zip(*results)

            return np.concatenate(results1), np.concatenate(results2), np.concatenate(results3), np.concatenate(
                results4), np.concatenate(results5), np.concatenate(results6)
        else:
            results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                         simulationChunk, itertools.repeat(env),
                                                         itertools.repeat(pC1E1Dict), itertools.repeat(adopt),
                                                         itertools.repeat(T), itertools.repeat(adoptionType)))
            pool.close()
            pool.join()

            results1, results2, results3 = zip(*results)
            return np.concatenate(results1), np.concatenate(
                results2), np.concatenate(
                results3)
            # results 2: the first dimension refers to agents, the second to
            # phenotypes and the third to time


    else:

        resultsAllENV = {}
        for cueRel in allENV:
            pC1E1Dict = pickle.load(
                open(os.path.join(resultsPath,
                                  'runTest_%s%s_%s_%s/pC1E1dict.p' % (argumentR[0], argumentP[0], pE1, cueRel)),
                     "rb"))
            print cueRel

            pool = Pool(processes=12)

            if adopt:

                results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                             simulationChunk, itertools.repeat(env),
                                                             itertools.repeat(pC1E1Dict), itertools.repeat(adopt),
                                                             itertools.repeat(T), itertools.repeat(adoptionType)))
                pool.close()
                pool.join()

                results1, results2 = zip(*results)
                allResults = np.concatenate(results1), np.concatenate(results2)
                resultsAllENV[cueRel] = allResults[0]
            else:
                results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                             simulationChunk, itertools.repeat(env),
                                                             itertools.repeat(pC1E1Dict), itertools.repeat(adopt),
                                                             itertools.repeat(T), itertools.repeat(adoptionType)))
                pool.close()
                pool.join()

                results1, results2 = zip(*results)
                return np.concatenate(results1), np.concatenate(
                    results2)  # the first dimension refers to agents, the second to
                # phenotypes and the third to time

        return resultsAllENV


def calcEuclideanDistance(original, doppelgaenger):
    result = [np.sqrt(np.sum((x - y) ** 2)) for x, y in zip(original[:, 0:2], doppelgaenger[:, 0:2])]
    return np.array(result)


def runExperimentalAdoptionExperiment(T, numAgents, env, prior, cueReliability, resultsPath, argumentR, argumentP, lag,
                                      adoptionType, endOfExposure):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # proportiional distance: absolute distance divided by maximum possible distance
    # maximum possible distance: 20 * sqrt(2)
    tValues = np.arange(1, T + 1, 1)
    resultLen = int(math.ceil(float(numAgents) / 12)) * 12
    results = np.zeros((T, resultLen))

    for t in tValues:
        print "currently working on time step: " + str(t)
        original, doppelgaenger = runExperimentalTwinStudiesParallel(t, numAgents, env, prior, cueReliability, lag, T,
                                                                     resultsPath, argumentR, argumentP, adoptionType,
                                                                     endOfExposure)
        results[t - 1, :] = calcEuclideanDistance(original, doppelgaenger)

    return results


def postProcessPosteriorBelief(posBeliefOrg, posBeliefDG):
    # first calculate difference across separation time; that is the average of the difference matrix
    absDifferencesOrg = np.absolute(np.diff(posBeliefOrg))
    meanDifferenceOrg = absDifferencesOrg.mean(axis=0)
    absDifferencesDG = np.absolute(np.diff(posBeliefDG))
    meanDifferenceDG = absDifferencesDG.mean(axis=0)
    posDifferences = np.abs(posBeliefOrg[:, 1:] - posBeliefDG[:, 1:])
    meanPosDifferences = posDifferences.mean(axis=0)
    return meanDifferenceOrg, meanDifferenceDG, meanPosDifferences

def runAdoptionExperiment(T, numAgents, env, prior, cueReliability, resultsPath, argumentR, argumentP, adoptionType):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # proportional distance: absolute distance divided by maximum possible distance
    # maximum possible distance: 20 * sqrt(2)
    tValues = np.arange(1, T + 1, 1)
    resultLen = int(math.ceil(float(numAgents) / 12)) * 12
    results = np.zeros((T, resultLen))
    resultsTempPhenotypes = np.zeros(
        (T, resultLen))  # euclidean distance between original and twin right after exposure
    resultsBeliefAggr = np.zeros((T, 3))

    posBeliefDiffStart = [0] * T
    posBeliefDiffEnd = [0] * T

    for t in tValues:
        print "currently working on time step: " + str(t)
        original, doppelgaenger, posBeliefOrg, posBeliefDG, originalTemp, doppelgaengerTemp = runTwinStudiesParallel(t,
                                                                                                                     numAgents,
                                                                                                                     env,
                                                                                                                     prior,
                                                                                                                     cueReliability,
                                                                                                                     True,
                                                                                                                     T,
                                                                                                                     resultsPath,
                                                                                                                     argumentR,
                                                                                                                     argumentP,
                                                                                                                     adoptionType,
                                                                                                                     [])

        if t == 1:
            simNum = posBeliefOrg.shape[0]
            posBeliefOrg[:, 0] = [1 - prior] * simNum
            posBeliefDG[:, 0] = [1 - prior] * simNum

        results[t - 1, :] = calcEuclideanDistance(original, doppelgaenger)
        resultsTempPhenotypes[t - 1, :] = calcEuclideanDistance(originalTemp, doppelgaengerTemp)
        meanDifferenceOrg, meanDifferenceDG, meanPosDifferences = postProcessPosteriorBelief(posBeliefOrg, posBeliefDG)
        if t == 1:
            posBeliefDeltaOrg = meanDifferenceOrg
            posBeliefDeltaOrg = posBeliefDeltaOrg.reshape(T, 1)
            posBeliefDeltaDG = meanDifferenceDG
            posBeliefDeltaDG = posBeliefDeltaDG.reshape(T, 1)
        posBeliefDiffEnd[t - 1] = meanPosDifferences[-1]  # TODO change store the last difference
        posBeliefDiffStart[t - 1] = meanPosDifferences[0]  # TODO change store the last difference

        # it might still be interesting to have a plot with one line per time of adoption indicating
        # belief change of the orginal in one plot, the doppelgaenger, and the posterior belief change

        # is the absolute average difference across time and agents in posterior belief interesting?
        # I think it might be: it is a different proxy for plasticity in belief
        # how different is twins' belief in environment 1 due to exposure to cues?; focus on this for now, but keep
        # thinking about this
        resultsBeliefAggr[t - 1, :] = [np.mean(meanDifferenceOrg), np.mean(meanDifferenceDG),
                                       np.mean(meanPosDifferences)]

    # need to add the other two columns
    posBeliefDiffEnd = np.array(posBeliefDiffEnd).reshape(T, 1)
    posBeliefDiffStart = np.array(posBeliefDiffStart).reshape(T, 1)

    resultsBeliefAggr = np.hstack((resultsBeliefAggr, posBeliefDeltaOrg, posBeliefDeltaDG, posBeliefDiffEnd, posBeliefDiffStart))
    return results, resultsBeliefAggr, resultsTempPhenotypes


def normRows(vec):
    if vec.min() != vec.max():
        curRowNorm = (vec - vec.min()) / float((vec.max() - vec.min()))
        return curRowNorm
    else:
        return vec

def meanAbsDistance(data, currMean):
    dataDiff = data-currMean
    return np.mean(abs(dataDiff))

def postProcessResultsMat(results, T, endOfExposure, lag):
    resultsVec = []
    resultsVecVar = []
    resultsVecRelative = []
    resultsVecRelativeVar = []

    if not endOfExposure:  # if phenotypic distance has been measured at the end of ontogeny
        resultsNorm = results / float(T * np.sqrt(2))
        for idx in range(results.shape[0]):
            curRowNorm = resultsNorm[idx, :]
            curRow = results[idx, :]
            curRowRelative = curRow / float((T - idx) * np.sqrt(2))

            resultsVec.append(np.mean(curRowNorm))
            resultsVecRelative.append(np.mean(curRowRelative))
            varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1])
            varAbs = meanAbsDistance(curRowNorm, resultsVec[-1])
            resultsVecVar.append(varAbs)
            resultsVecRelativeVar.append(varRel)
    else:
        for idx in range(results.shape[0]):
            curRow = results[idx, :]
            curRowNorm = curRow / float((lag + idx) * np.sqrt(2))
            curRowRelative = curRow / float(lag * np.sqrt(2))
            resultsVec.append(np.mean(curRowNorm))
            resultsVecRelative.append(np.mean(curRowRelative))
            varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1])
            varAbs = meanAbsDistance(curRowNorm, resultsVec[-1])
            resultsVecVar.append(varAbs)
            resultsVecRelativeVar.append(varRel)
    return resultsVec, resultsVecRelative, resultsVecVar, resultsVecRelativeVar


def rescaleNumbers(newMin, newMax, numbersArray):
    OldMin = np.min(numbersArray)
    OldMax = np.max(numbersArray)
    result = [(((OldValue - OldMin) * (newMax - newMin)) / float(OldMax - OldMin)) + newMin for OldValue in
              numbersArray]
    return result


def area_calc(probs, r):
    # result = [(p)**2 * np.pi*r for p in probs]
    result = [np.sqrt(float(p)) * r for p in probs]
    return result



def plotTriangularPlots(tValues, priorE0Arr, cueValidityArr, maturePhenotypes, T, twinResultsPath):
    # first step is to permute indices
    permuteIdx = [0, 2, 1]

    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            plt.sca(ax)

            """
            Here goes the actual plotting code 
            """
            maturePhenotypesCurr = maturePhenotypes[(pE0, cueVal)]
            numAgents = maturePhenotypesCurr.shape[0]
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
            # now need to work on the scaling of points
            unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
            # area = area_calc(uniqueCounts / float(numAgents), 150)
            area2 = (uniqueCounts / float(numAgents)) * 250
            # this one would be scalling according to area
            tax.scatter(unique[:, permuteIdx], s=area2, color='grey')
            tax.boundary(linewidth=0.8, zorder=-1)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            if jx == len(priorE0Arr) / 2 and ix == len(cueValidityArr) / 2:
                fontsize = 20
                tax.right_corner_label("P0", fontsize=fontsize)
                tax.top_corner_label("wait time", fontsize=fontsize)
                tax.left_corner_label("P1", fontsize=fontsize)
                tax._redraw_labels()
            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'ternary.png'), dpi=1200)
    plt.close()


def fitnessDifference(pE0Arr, cueValidityArr, policyPath, T, resultsPath, baselineFitness, argumentR, argumentP):
    # fitness functions
    # keep in mind fitnessMax is equivalent to T
    beta = 0.2
    if argumentR == 'linear':
        phiVar = lambda y: y

    elif argumentR == 'diminishing':
        alphaRD = (T) / float(1 - float(np.exp(-beta * (T))))
        phiVar = lambda y: alphaRD * (1 - np.exp(-beta * y))

    elif argumentR == 'increasing':
        alphaRI = (T) / float(float(np.exp(beta * (T))) - 1)
        phiVar = lambda y: alphaRI * (np.exp(beta * y) - 1)
    else:
        print 'Wrong input argument to additive fitness reward function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    if argumentP == 'linear':
        psiVar = lambda y: -(y)

    elif argumentP == 'diminishing':
        alphaPD = (T) / float(1 - float(np.exp(-beta * (T))))
        psiVar = lambda y: -(alphaPD * (1 - np.exp(-beta * y)))

    elif argumentP == 'increasing':

        alphaPI = (T) / float(float(np.exp(beta * (T))) - 1)
        psiVar = lambda y: -(alphaPI * (np.exp(beta * y) - 1))


    else:
        print 'Wrong input argument to additive fitness penalty function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    # dictionary for storing the results
    resultsDict = {}
    for priorE0 in pE0Arr:
        for cueReliability in cueValidityArr:

            prior = 1 - priorE0
            print "Currently calculating expected fitness differences with prior: " + str(
                prior) + " and cue reliability: " + str(cueReliability)

            # fitness following the optimal policy
            # simulate mature phenotypes for each environment
            maturePhenotypesEnv1 = runTwinStudiesParallel(0, 5000, 1, priorE0, cueReliability, False, T, policyPath,
                                                          argumentR, argumentP, None, [])[0]
            maturePhenotypesEnv0 = runTwinStudiesParallel(0, 5000, 0, priorE0, cueReliability, False, T, policyPath,
                                                          argumentR, argumentP, None, [])[0]

            OEnv1 = np.mean(np.array([phiVar(y[1]) + psiVar(y[0]) for y in maturePhenotypesEnv1]))
            OEnv0 = np.mean(np.array([phiVar(y[0]) + psiVar(y[1]) for y in maturePhenotypesEnv0]))
            OFitness = ((prior * OEnv1 + (1 - prior) * OEnv0) - baselineFitness) / float(T)  # scale to a range of 1

            # next specialist Fitness
            if prior > 0.5:
                phenotypeS = np.array([0, T, 0])
                SEnv1 = phiVar(phenotypeS[1]) + psiVar(phenotypeS[0])
                SEnv0 = phiVar(phenotypeS[0]) + psiVar(phenotypeS[1])

            else:
                specialistPhenotypes = np.zeros((5000, 2))
                specialistPhenotypes[:, 0] = np.append(np.array([T] * 2500), np.array([0] * 2500))
                specialistPhenotypes[:, 1] = np.append(np.array([0] * 2500), np.array([T] * 2500))

                SEnv1 = np.mean(np.array([phiVar(y[1]) + psiVar(y[0]) for y in specialistPhenotypes]))
                SEnv0 = np.mean(np.array([phiVar(y[0]) + psiVar(y[1]) for y in specialistPhenotypes]))

            SFitness = ((prior * SEnv1 + (1 - prior) * SEnv0) - baselineFitness) / float(T)
            phenotypeG = np.array([T / float(2), T / float(2)])
            GEnv1 = phiVar(phenotypeG[1]) + psiVar(phenotypeG[0])
            GEnv0 = phiVar(phenotypeG[0]) + psiVar(phenotypeG[1])
            GFitness = ((prior * GEnv1 + (1 - prior) * GEnv0) - baselineFitness) / float(T)

            resultsDict[(priorE0, cueReliability)] = np.array([SFitness, OFitness, GFitness])

    pickle.dump(resultsDict, open(os.path.join(resultsPath, "fitnessDifferences.p"), "wb"))


def plotFitnessDifference(priorE0Arr, cueValidityArr, twinResultsPath):
    # first open the dictionary containing the results

    differencesDict = pickle.load(open(os.path.join(twinResultsPath, "fitnessDifferences.p"), "rb"))
    # define the xAxis
    x = np.arange(3)
    xLabels = ["S", "O", "G"]
    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            plt.sca(ax)
            # open the relevant fitness difference array
            fitnessDifferences = differencesDict[(pE0, cueVal)]

            barList = plt.bar(x, fitnessDifferences)

            barList[0].set_color("lightgray")
            barList[1].set_color("grey")
            barList[2].set_color("black")

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=20, labelpad=10)
                plt.xticks(x, xLabels, fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('Fitness difference', fontsize=20, labelpad=10)
                plt.yticks([-1, 0, 1], fontsize=15)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1

        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'fitnessDifferences.png'), dpi=1200)
    plt.close()


def calcNegativeRankSwitches(rankDf, T, arg):
    tValues = np.arange(0, T, 1)
    results = np.zeros((T, T))
    # need possible number of ranks at each time step
    for t in tValues:
        rankDfDiff = rankDf.loc[:, t:].sub(rankDf.loc[:, t], axis=0)
        rankDfDiff2 = rankDfDiff.copy(deep=True)
        if arg == 'unstable':
            rankDfDiff[rankDfDiff2 == 0] = 0
            rankDfDiff[rankDfDiff2 != 0] = 1
        else:
            rankDfDiff[rankDfDiff2 != 0] = 0
            rankDfDiff[rankDfDiff2 == 0] = 1
        results[t, t:] = rankDfDiff.sum(axis=0) / float(rankDf.shape[0])

    return results


def plotRankOrderStability(priorE0Arr, cueValidityArr, twinResultsPath, T, types):
    for distFun in types:
        plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun)


def createLABELS(T):
    labels = [" "] * T
    labels[0] = str(1)
    labels[T - 1] = str(T)
    labels[int(T / 2) - 1] = str(T / 2)
    return labels


def plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun):
    """
    :param priorE0Arr:
    :param cueValidityArr:
    :param twinResultsPath:
    :param T:
    :param distFun:
    :return:
    """

    '''
    We cannot use a correlation coefficient to determine rank-order stability because there might be cases in which 
    there is no variability in ranks 
    '''

    # first open the dictionary containing the results
    # for prior, cue reliability combination it contains a matrix with the ranks across time steps
    ranks = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRanks.p"), "rb"))

    # what do we want to plot?
    # could have a plot with the correlation coefficient between consecutive timesteps
    # or a whole correlation matrix, heatplot? start with this
    # want to represent the proportion of ties as well

    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    plt.subplots_adjust(top =0.92, bottom = 0.12)
    specialAx = fig.add_axes([.16, .040, .7, .01])
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for pE0 in priorE0Arr:
            rankMatrix = ranks[(pE0, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1

            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')

            simRange += list(sim.flatten())

    boundary1 = min(simRange)
    boundary2 = max(simRange)

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            rankMatrix = ranks[(pE0, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
            # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
                axis=0)] + 0.1  # returns columns that are all zeros

            # calculating the similarity matrix
            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
                cmap = 'YlGnBu'
                yLabel = 'Cosine similarity'
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
                cmap = 'Greys'  # 'YlGnBu'
                yLabel = 'ontogeny'

            # only negative rank switches

            # create a mask for the upper triangle
            mask = np.tri(sim.shape[0], k=0)




            if jx == len(priorE0Arr) - 1 and ix == 0:
                cbar = True
                cbar_ax = specialAx
                cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
                            'label': "proportion of rank switches"}  # 'label':"Proportion of negative rank switches",
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)

                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=15)
                cbar.ax.xaxis.label.set_size(20)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)


            else:
                cbar = False
                cbar_ax = None
                cbar_kws = None
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
                ax.tick_params(labelsize=15)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax2.set_ylim(0, 1)
            ax2.get_xaxis().tick_bottom()
            ax2.get_yaxis().tick_right()

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.set_xlabel('ontogeny', fontsize=20, labelpad=15)
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)

                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.tick_params(labelsize=15)
            else:
                #ax.get_xaxis().set_visible(False)
                ax2.set_yticks([])

            # if jx == 0:
            #     ax.yaxis.set_label_position("left")
            #     ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)
            #     ax2.set_yticks(np.arange(0, 1.1, 0.2))
            #     ax2.tick_params(labelsize=15)
            # else:
            #     ax2.set_yticks([])

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStability2%s.png' % distFun), dpi=1200)
    plt.close()

    # # second plot is for rank stability
    # fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    # specialAx = fig.add_axes([.16, .055, .7, .01])
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, "stable")
    #
    #         simRange += list(sim.flatten())
    #
    # boundary1 = min(simRange)
    # boundary2 = max(simRange)
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #             cmap = 'YlGnBu'
    #             yLabel = 'Cosine similarity'
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'stable')
    #             cmap = 'Greys'  # 'YlGnBu'
    #             yLabel = 'Time step'
    #
    #         # only negative rank switches
    #
    #         # create a mask for the upper triangle
    #         mask = np.tri(sim.shape[0], k=0)
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             cbar = True
    #             cbar_ax = specialAx
    #             cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
    #                         'label': "Proportion of stable ranks"}  # 'label':"Proportion of negative rank switches",
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #
    #             cbar = ax.collections[0].colorbar
    #             # here set the labelsize by 20
    #             cbar.ax.tick_params(labelsize=14)
    #             cbar.ax.xaxis.label.set_size(20)
    #         else:
    #             cbar = False
    #             cbar_ax = None
    #             cbar_kws = None
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #             ax.tick_params(labelsize=14)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.spines['left'].set_visible(False)
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=10)
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos1%s.png' % distFun), dpi=1200)
    # plt.close()
    #
    # # 3rd plot
    # fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         simRange += list(sim.flatten())
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)
    #
    #
    #         else:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(True)
    #         ax.spines['left'].set_visible(False)
    #
    #         ax.set_ylim(0, 1)
    #         plt.yticks([])
    #         plt.xticks([])
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #         #
    #         # if jx == 0:
    #         #     plt.title(str(cueVal), fontsize=30)
    #         #
    #         # if ix == 0 and jx == 0:
    #         #     ax.set_xlabel('Time', fontsize=30, labelpad=10)
    #         #     ax.spines['left'].set_visible(True)
    #         #     ax.yaxis.set_label_position("left")
    #         #     ax.set_ylabel('Proportion of rank switches', fontsize=30, labelpad=10)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('Proportion of rank switches', fontsize=20, labelpad=10)
    #
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos2%s.png' % distFun), dpi=1200)
    # plt.close()




def plotBeliefAndPhenotypeDivergence(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            relativeDistanceDiff = np.gradient(relativeDistance)


            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:, 5] #measured at the end of ontogeny after the last cue
            posBeliefDiffNoAverageDiff = np.gradient(posBeliefDiffNoAverage)


            plt.plot(tValues[0:], posBeliefDiffNoAverageDiff, color='grey', linestyle='solid', linewidth=2, markersize=5,
                         marker='D',
                         markerfacecolor='grey')

            plt.plot(tValues[0:], relativeDistanceDiff, color='black', linestyle='solid', linewidth=2, markersize=5,
                         marker='o',
                         markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.4, 0.05)
            plt.yticks([-0.3,0,0.05], fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('gradient of plasticity curves', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues)-0.5,max(tValues)+0.5+1,1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefEndOntogenyDivergence.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()


def plotBeliefDistances(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:, 5] #measured at the end of ontogeny after the last cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            plt.plot(tValues, relativeDistance, color='black', linestyle='solid', linewidth=2, markersize=8,
                         marker='o',
                         markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)


            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues)-0.5,max(tValues)+0.5+1,1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefEndOntogeny.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()

    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:,
                                         6]  # measured after each cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            relativeDistanceTemp = relativeDistanceDictTemp[(pE0, cueVal)]
            plt.plot(tValues, relativeDistanceTemp, color='black', linestyle='solid', linewidth=2, markersize=8,
                         marker='o',markerfacecolor='black')

            print "The current prior is %s and the cue reliability is %s" % ((1 - pE0), cueVal)
            print "The correlation between information and phenotype divergence is: " + str(
                stats.pearsonr(relativeDistanceTemp, posBeliefDiffNoAverage)[0])


            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)


            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)


            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability
            tValNew = np.arange(min(tValues) - 0.5, max(tValues) + 0.5 + 1, 1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefAfterCue.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()


def plotDistances(tValues, priorE0Arr, cueValidityArr, absoluteDistanceDict, relativeDistanceDict, twinResultsPath,
                  argument, adoptionType, lag, endOfExposure, VarArg, absoluteDistanceDictVar, relativeDistanceDictVar):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)
            absoluteDistance = absoluteDistanceDict[(pE0, cueVal)]
            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            if VarArg:

                absoluteDistanceVar = absoluteDistanceDictVar[(pE0, cueVal)]
                plt.plot(tValues, absoluteDistance, color='grey', linestyle='solid', linewidth=2, markersize=8,
                        marker='D',
                        markerfacecolor='grey')
                plt.errorbar(tValues, absoluteDistance, yerr=absoluteDistanceVar, fmt="none", ecolor='grey')

                relativeDistanceVar = relativeDistanceDictVar[(pE0, cueVal)]

                plt.plot(tValues, relativeDistance, color='black', linestyle='--', linewidth=2, markersize=8,
                         marker='o', markerfacecolor='black')
                plt.errorbar(tValues, relativeDistance,yerr = relativeDistanceVar, fmt ="none", ecolor= 'grey')


            else:
                plt.plot(tValues, absoluteDistance, color='grey', linestyle='solid', linewidth=2, markersize=8,
                         marker='D',
                         markerfacecolor='grey')
                plt.plot(tValues, relativeDistance, color='black', linestyle='solid', linewidth=2, markersize=8,
                         marker='o', markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)




            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0,1.1,0.2), fontsize = 15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)


            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('phenotypic distance', fontsize=20, labelpad=10)


            elif jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")


            # plot lines for readeability
            plt.plot(tValues,[1]*len(tValues), ls = '--', lw = 0.8, color = '#B8B8B8', zorder = -1)
            plt.plot(tValues, [0] * len(tValues), ls='--', lw=0.8, color='#B8B8B8', zorder = -1)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticity%s.png' % (argument, adoptionType, lag, safeStr, VarArg)),
        dpi=1200)
    plt.close()



def performSimulationAnalysis(argument, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType, endOfExposure):
    # first step create the directory for the results
    if not os.path.exists(twinResultsPath):
        os.makedirs(twinResultsPath)

    if argument == "ExperimentalTwinstudy":
        """
        this will implement a form of the twin study that an be considered more artifical
        it will be comparable to experimental manipulations done in lab environments
        it will manipulate the onset and amount of a
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}

        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "currently working with prior: " + str(prior) + " and cue reliability: " + str(cueReliability)

                startTime = time.time()
                resultsMat = runExperimentalAdoptionExperiment(T, numAgents, 1, prior, cueReliability, resultsPath,
                                                               argumentR, argumentP, lag, adoptionType, endOfExposure)
                elapsedTime = time.time() - startTime
                print "Elapsed time: " + str(elapsedTime)

                # normalize resultsmat
                pickle.dump(resultsMat,
                            open(os.path.join(twinResultsPath, "resultsMat_%s_%s.p" % (prior, cueReliability)), "wb"))
                absoluteDistance, relativeDistance, _,_ = postProcessResultsMat(resultsMat, T + lag - 1, endOfExposure, lag)
                absoluteDistanceDict[(prior, cueReliability)] = absoluteDistance
                relativeDistanceDict[(prior, cueReliability)] = relativeDistance

        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))

        # plasticityAreaGradient(priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath)

    elif argument == "Twinstudy":
        """
        This will calculate the results from the twin studies
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}
        absoluteDistanceDictVar = {}
        relativeDistanceDictVar = {}

        absoluteDistanceDictTemp = {}
        relativeDistanceDictTemp = {}

        beliefDict = {}

        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "currently working with prior: " + str(prior) + " and cue reliability: " + str(cueReliability)

                startTime = time.time()
                resultsMat, resultsMatBeliefs, resultsMatTempPhenotypes = runAdoptionExperiment(T, numAgents, 1, prior,
                                                                                                cueReliability,
                                                                                                resultsPath, argumentR,
                                                                                                argumentP, adoptionType)
                elapsedTime = time.time() - startTime
                print "Elapsed time: " + str(elapsedTime)

                # normalize resultsmat
                pickle.dump(resultsMat,
                            open(os.path.join(twinResultsPath, "resultsMat_%s_%s.p" % (prior, cueReliability)), "wb"))

                absoluteDistance, relativeDistance, absoluteDistanceVariance, relativeDistanceVariance = postProcessResultsMat(
                    resultsMat, T, endOfExposure, lag)

                absoluteDistanceDict[(prior, cueReliability)] = absoluteDistance
                relativeDistanceDict[(prior, cueReliability)] = relativeDistance

                absoluteDistanceDictVar[(prior, cueReliability)] = absoluteDistanceVariance
                relativeDistanceDictVar[(prior, cueReliability)] = relativeDistanceVariance

                beliefDict[(prior, cueReliability)] = resultsMatBeliefs

                # do the same for the temporary phenotypes
                pickle.dump(resultsMatTempPhenotypes,
                            open(os.path.join(twinResultsPath,
                                              "resultsMatTempPhenotypes_%s_%s.p" % (prior, cueReliability)), "wb"))
                absoluteDistanceTemp, relativeDistanceTemp,_,_ = postProcessResultsMat(resultsMatTempPhenotypes, T, True, 1)
                absoluteDistanceDictTemp[(prior, cueReliability)] = absoluteDistanceTemp
                relativeDistanceDictTemp[(prior, cueReliability)] = relativeDistanceTemp

        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))

        pickle.dump(absoluteDistanceDictVar, open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))
        pickle.dump(relativeDistanceDictVar, open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))

        pickle.dump(absoluteDistanceDictTemp,
                    open(os.path.join(twinResultsPath, 'absoluteDistanceDictTemp%s%s%s%s.p' % (
                        argument, adoptionType, lag, endOfExposure)), 'wb'))
        pickle.dump(relativeDistanceDictTemp,
                    open(os.path.join(twinResultsPath, 'relativeDistanceDictTemp%s%s%s%s.p' % (
                        argument, adoptionType, lag, endOfExposure)), 'wb'))

        pickle.dump(beliefDict, open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'wb'))

        # plasticityAreaGradient(priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath)

    elif argument == "reactionNorm":
        maturePhenotypesRN = {}
        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on prior: " + str(prior) + " and cue reliability: " + str(cueReliability)
                # removed taking the 0th element from the result list, keep that in mind for plotting processing
                maturePhenotypesRN[(prior, cueReliability)] = runTwinStudiesParallel(0, numAgents, 1, prior,
                                                                                     cueReliability, False, T,
                                                                                     resultsPath, argumentR, argumentP,
                                                                                     None, cueValidityC0E0Arr)

        pickle.dump(maturePhenotypesRN, open(os.path.join(twinResultsPath, "maturePhenotypesRN.p"), "wb"))

    elif argument == "MaturePhenotypes":
        maturePhenotypes = {}
        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on prior: " + str(prior) + " and cue reliability: " + str(cueReliability)
                maturePhenotypes[(prior, cueReliability)] = runTwinStudiesParallel(0, numAgents, 1, prior,
                                                                                   cueReliability, False, T,
                                                                                   resultsPath, argumentR, argumentP,
                                                                                   None, [])[0]

        pickle.dump(maturePhenotypes, open(os.path.join(twinResultsPath, "maturePhenotypes.p"), "wb"))

    elif argument == "RankOrderStability":
        rankOrderStabilityRaw = {}
        rankOrderStabilityRanks = {}
        rankOrderStabilityRanksNorm = {}

        # rankOrderStabilityRaw = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRaw.p"), "rb"))

        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on prior: " + str(prior) + " and cue reliability: " + str(cueReliability)
                rankOrderStabilityRaw[(prior, cueReliability)] = runTwinStudiesParallel(0, numAgents, 1, prior,
                                                                                        cueReliability, False, T,
                                                                                        resultsPath, argumentR,
                                                                                        argumentP,
                                                                                        None, [])[1]

                current = rankOrderStabilityRaw[(prior, cueReliability)]
                currentMat = np.zeros((current.shape[0], T))
                currentMatNorm = np.zeros((current.shape[0], T))

                tValues = np.arange(1, T + 1, 1)
                for t in tValues:
                    possibleRanks = sorted(list(set(current[:, 1, t - 1])), reverse=True)
                    currentMat[:, t - 1] = [possibleRanks.index(a) + 1 for a in current[:, 1,
                                                                                t - 1]]  # the plus one makes sure that we don't have zero ranks, which are computationally inconvenient

                rankOrderStabilityRanks[(prior, cueReliability)] = currentMat
                rankOrderStabilityRanksNorm[(prior, cueReliability)] = currentMatNorm

        pickle.dump(rankOrderStabilityRaw, open(os.path.join(twinResultsPath, "rankOrderStabilityRaw.p"), "wb"))
        pickle.dump(rankOrderStabilityRanks, open(os.path.join(twinResultsPath, "rankOrderStabilityRanks.p"), "wb"))


    elif argument == "MaturePhenotypesTwoPatches":

        maturePhenotypes = {}
        for prior in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on prior: " + str(prior) + " and cue reliability: " + str(cueReliability)
                agentsEnv1 = int((1 - prior) * numAgents)
                agentsEnv0 = numAgents - agentsEnv1
                resultsEnv1 = runTwinStudiesParallel(0, agentsEnv1, 1, prior,
                                                     cueReliability, False, T,
                                                     resultsPath, argumentR, argumentP, None, [])[0]
                resultsEnv0 = runTwinStudiesParallel(0, agentsEnv0, 0, prior,
                                                     cueReliability, False, T,
                                                     resultsPath, argumentR, argumentP, None, [])[0]

                maturePhenotypes[(prior, cueReliability)] = np.concatenate([resultsEnv1, resultsEnv0])
        pickle.dump(maturePhenotypes, open(os.path.join(twinResultsPath, "maturePhenotypesTwoPatches.p"), "wb"))



    elif argument == 'FitnessDifference':
        fitnessDifference(priorE0Arr, cueValidityC0E0Arr, resultsPath, T, twinResultsPath, baselineFitness, argumentR,
                          argumentP)


    else:
        print "Wrong input argument to plotting arguments!"


def plotSimulationStudy(argument, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType, endOfExposure, varArg):
    tValues = np.arange(1, T + 1, 1)

    if argument == "BeliefTwinstudy":
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure)), 'rb'))

        # for the temporary phenotypes
        relativeDistanceDictTemp = pickle.load(
            open(os.path.join(twinResultsPath, 'relativeDistanceDictTemp%s%s%s%s.p' % (
                "Twinstudy", adoptionType, lag, endOfExposure)), 'rb'))

        beliefDict = pickle.load(open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure)), 'rb'))

        plotBeliefDistances(tValues, priorE0Arr, cueValidityC0E0Arr, relativeDistanceDict, twinResultsPath,
                            argument, adoptionType, lag, endOfExposure, beliefDict,
                            relativeDistanceDictTemp)
        plotBeliefAndPhenotypeDivergence(tValues, priorE0Arr, cueValidityC0E0Arr, relativeDistanceDict, twinResultsPath,
                            argument, adoptionType, lag, endOfExposure, beliefDict,
                            relativeDistanceDictTemp)

    elif argument == "Twinstudy":

        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'rb'))

        # load the variance
        if varArg:
            absoluteDistanceDictVar = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s.p' % (
                argument, adoptionType, lag, endOfExposure)), 'rb'))
            relativeDistanceDictVar = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s.p' % (
                argument, adoptionType, lag, endOfExposure)), 'rb'))

        else:
            absoluteDistanceDictVar = None
            relativeDistanceDictVar = None
        plotDistances(tValues, priorE0Arr, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, varArg, absoluteDistanceDictVar,
                      relativeDistanceDictVar)

    elif argument == "ExperimentalTwinstudy":
        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s.p' % (
            argument, adoptionType, lag, endOfExposure)), 'rb'))

        plotDistances(tValues, priorE0Arr, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, False, None, None)

    elif argument == "MaturePhenotypes":
        maturePhenotypes = pickle.load(open(os.path.join(twinResultsPath, "maturePhenotypes.p"), "rb"))
        plotTriangularPlots(tValues, priorE0Arr, cueValidityC0E0Arr, maturePhenotypes, T, twinResultsPath)

    elif argument == "MaturePhenotypesTwoPatches":
        maturePhenotypes = pickle.load(open(os.path.join(twinResultsPath, "maturePhenotypesTwoPatches.p"), "rb"))
        plotTriangularPlots(tValues, priorE0Arr, cueValidityC0E0Arr, maturePhenotypes, T, twinResultsPath)

    elif argument == "FitnessDifference":
        plotFitnessDifference(priorE0Arr, cueValidityC0E0Arr, twinResultsPath)

    elif argument == "RankOrderStability":
        plotRankOrderStability(priorE0Arr, cueValidityC0E0Arr, twinResultsPath, T, ['negativeSwitches'])

    else:
        print "Wrong input argument to plotting arguments!"


def runPlots(priorE0Arr, cueValidityC0E0Arr, TParam, numAgents, twinResultsPath, baselineFitness, resultsPath,
             argumentR, argumentP, lagArray, adoptionType, endOfExposure, plotArgs, plotVar,performSimulation):
    for arg in plotArgs:
        print arg

        if arg == 'ExperimentalTwinstudy':
            for lag in lagArray:
                TLag = TParam - lag + 1
                T = TLag
                if performSimulation:
                    performSimulationAnalysis(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType,
                                              endOfExposure)
                plotSimulationStudy(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType,
                                    endOfExposure, plotVar)

        else:
            T = TParam
            if performSimulation:
                performSimulationAnalysis(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                                          resultsPath, baselineFitness, argumentR, argumentP, None, adoptionType,
                                          False)
            plotSimulationStudy(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, None, adoptionType,
                                False, plotVar)
