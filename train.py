#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from scipy.stats import binom, rankdata
from multiprocessing import Pool
import statsmodels.stats.multitest as smm
import sys
import argparse

def tuneModel(Y, X, cores, base):
    trainRange = np.arange(3,40)
    clf = RandomForestClassifier(bootstrap=True,n_jobs=cores)
    #train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X,Y,test_size=0.2)
    loocv = cross_validation.LeaveOneOut(len(X))
    parameters = {'n_estimators':trainRange}
    cv = GridSearchCV(clf, param_grid=parameters, cv = loocv, n_jobs=cores)
    finalModel = cv.fit(X,Y)
    sys.stderr.write('Trainned random forest for %s!\n' %base)
    return finalModel

def subsetData(df,base):
    #define features and labels
    features = ['A','C','T','G','deletion']
    features.remove(base) 
    dfPredictors = df[features].values
    rowsum = np.sum(dfPredictors,axis=1)
    X = np.true_divide(dfPredictors.T  ,rowsum).T
    return X
pd.DataFrame.subsetData = subsetData

def classifications(args):
    trainDf, cores, testDf, base = args
    sys.stderr.write('Filtering %s for training\n' %(base))
    subTrainDf = trainDf[trainDf['ref']==base]
    subTestDf = testDf[testDf['ref']==base]
    train_X = subTrainDf.subsetData(base)
    testX = testDf.subsetData(base)
    tunedModel = tuneModel(subTrainDf['abbrev'].values, train_X, cores, base)
    subTestDf['label'] = tunedModel.predict(newX)
    return subTestDf

def getPvalue(correctReads,cov,seqErr):
    assert len(correctReads)==len(cov), 'Wrong length of vector!'
    ps = binom.cdf(correctReads,n=cov,p=1-seqErr)
    return ps

def filterData(df,hyp,seqErr,pThreshold):
    mismatchBases = df[['A','C','T','G']].values
    df['mismatch'] = np.sum(mismatchBases,axis=1)
    df['refRead'] = df['cov'] - df['mismatch']
    if hyp == 'hyp1':
        df['p'] = getPvalue(df['refRead'].values,df['cov'].values,seqErr)
    elif hyp == 'hyp2':
        df['secondRef'] = np.amax(mismatchBases,axis=1)
        df['correctHeteroReads'] = np.sum(df[['refRead','secondRef']],axis=1)
        df['p'] = getPvalue(df['correctHeteroReads'].values,df['cov'].values,seqErr)
    df['padj'] =  smm.multipletests(df['p'].values,alpha=0.05,method='fdr_bh')[1]
    df = df[(df['padj']<pThreshold) & (df['cov']>10)]
    return df
pd.DataFrame.filterData = filterData

def main():
#    hyp = 'hyp2'
#    seqErr = 0.015
#    pThreshold = 0.05
#    cores = 20
#    tablename = 'teiTable.tsv'
    #get variables
    parser = argparse.ArgumentParser(description='Predicting modifications from mismatches')
    parser.add_argument('--hyp', default='hyp1',
            help = 'hypothesis to use: hyp1|hyp2 (default: hyp1)')
    parser.add_argument('--pval', default=0.05,type=float,
            help = 'FDR threshold [0-1] default: 0.05')
    parser.add_argument('--seqErr', default=0.015,type=float,
            help = 'sequencing PCR error rate [0-1] default : 0.015')
    parser.add_argument('--Enzyme',nargs='?', required=True, choices=['tei','gsi'],
            help = 'Enzyme used for the library: tei|gsi')
    parser.add_argument('--cores', default=1,type=int,
            help = 'cores number to use (default: 1)')
    parser.add_argument('--testSet',default='-',
            help = 'Sample bed file for classifying modifications for mismatch positions. [default: - (for stdin)]')
    parser.add_argument('--outfile', default='-', help='Outfile name, default: stdout (-)')
    args = parser.parse_args()
    
    # get in/out file names
    testTablename = args.testSet if args.testSet != '-' else sys.stdin
    outfilename = args.outfile if args.outfile != '-' else sys.stdout
    tablename = args.Enzyme+'Table.tsv'

    message = 'Starting prediction:\n' +\
            'Using parameters:\n' +\
            '\terror rate: %.3f\n' %(args.seqErr) +\
            '\tp-cutoff:   %.3f\n' %(args.pval)+\
            '\tcores:      %i\n' %(args.cores) +\
            '\tenzyme:     %s\n' %(args.Enzyme) +\
            '\thypothesis: %s\n' %(args.hyp)
    sys.stderr.write(message)

    #read files
    trainDf = pd.read_table(tablename,sep='\t') \
            .filterData(args.hyp, args.seqErr, args.pval)
    testDf = pd.read_table(testTablename,sep='\t') \
            .filterData(args.hyp, args.seqErr, args.pval)

    bases = np.unique(testDf['ref'].values)
    predictedData = map(classifications, [(trainDf, args.cores, testDf, base) for base in bases])
    predictedDF = pd.concat(predictedData,axis=0)
    predictedDF.to_csv(outfilename,sep='\t',index=False,
            columns = ['chrom','start','end','ref','cov','strand','label'])

if __name__ == '__main__':
    main()


