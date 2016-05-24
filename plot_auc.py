#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from scipy.stats import binom, rankdata
from multiprocessing import Pool
import statsmodels.stats.multitest as smm
import seaborn as sns
import matplotlib.pyplot as plt

clf = OneVsRestClassifier(RandomForestClassifier())

def subsetData(df,base):
    #define features and labels
    features = ['A','C','T','G','deletion']
    features.remove(base) 
    dfPredictors = df[features].values
    rowsum = np.sum(dfPredictors,axis=1)
    X = np.true_divide(dfPredictors.T  ,rowsum).T
    return np.array(X,dtype=np.float64)
pd.DataFrame.subsetData = subsetData

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

def mergeAbbrev(abb):
    if abb in ['m1A','m1I']:
        return 'm1A|m1I'
    elif abb in ['m2G','m2,2G']:
        return 'm2G|m2,2G'
    else:
        return abb
    
def modificationType(abb):
    if abb in ['m1A|m1I','I','m3C','acp3U','m3U','m2G|m2,2G','m1G','m6,6A']:
        return 'On basepair'
    else:
        return 'not on basepair'

def makeEnzyme(enz):
    return ['TeI-4c' if e == 'tei' else 'Gsi-IIc' for e in enz]

def get_roc(y_test, y_score, y_train):
    classes = np.unique(y_train)
    classes = np.append(classes,'null') #add null class to avoid true_binary model
    y = label_binarize(y_test, classes=classes)
    y = y[:,:-1]

    roc_auc = {}
    for i in range(len(classes[:-1])):
        multiclass = classes[i]
        fpr, tpr, _ = metrics.roc_curve(y[:, i], y_score[:, i])
        if len(tpr) > 2:
            roc_auc[multiclass] = metrics.auc(fpr, tpr)
    return roc_auc

def validation(X,Y):
    repeats = 20
    metric_list = [] 
    parameters = {'estimator__n_estimators':np.arange(5,40)}
        
    for i in np.arange(repeats):
        # split train test
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2,random_state=i)   
        loocv = cross_validation.LeaveOneOut(n = len(X_train))
        kfold = cross_validation.KFold(n = len(X_train),n_folds = 6)
        cv = GridSearchCV(clf,param_grid=parameters,n_jobs=-1, cv=loocv)
        cv.fit(X_train, y_train)
        tuned_model = cv.best_estimator_
        model = tuned_model.fit(X_train,y_train)
        y_score = model.predict_proba(X_test)
        roc_auc = get_roc(y_test, y_score, y_train)
        roc_df = pd.DataFrame(roc_auc, index=['auc'])
        roc_df = pd.melt(roc_df, value_name='AUC',var_name='classes')
        roc_df['repeat'] = np.repeat(i,len(roc_df))
        metric_list.append(roc_df)
        print 'Trained %i times' %i
    return pd.concat(metric_list,axis=0)
    
def trainBase(df, base):
    print 'Training base: %s' %base
    filtered_df = df[df['ref'] == base]
    classes = np.unique(filtered_df['abbrev'])
    if len(classes) != 1 and len(filtered_df)> 20:
        X = filtered_df.subsetData(base)
        Y = filtered_df['abbrev'].values
        res = validation(X,Y)
        res['base'] = np.repeat(base, len(res))
        return res
    else:
        print 'Skipping %s '%base

def readFile(enzyme):
    print 'Running enzyme %s' %enzyme
    filename = '../trainSets/' + enzyme + 'Table.json'
    trainDf = pd.read_json(filename,'table').filterData('hyp1', 0.05, 0.05)
    trainDf['abbrev'] = map(mergeAbbrev,trainDf.abbrev)

    auc = [trainBase(trainDf,base) for base in list('ACTG')]
    auc = pd.concat(auc,axis=0)
    auc['type'] = map(modificationType,auc['classes'])
    auc['enzyme'] = np.repeat(enzyme,len(auc))
    auc['enzyme'] = makeEnzyme(auc['enzyme'])
    return auc

def plot(aucDF, figurename):
    color_order = np.unique(aucDF['type'])
    sns.set_style('white')
    with sns.plotting_context('paper',font_scale=1.3):
        p = sns.FacetGrid(data = aucDF, col = 'base', row='enzyme',
                sharex=False, aspect = 1.5, size=3,margin_titles=True)
    p.map(sns.swarmplot, 'classes', 'AUC', 'type',
            hue_order = color_order,
            palette=sns.color_palette("hls", 2))
    p.add_legend()
    p.set(ylim=(0,1.2))
    p.set(xlabel=' ', ylabel=' ')
    [plt.setp(ax.texts, text="") for ax in p.axes.flat]
    [plt.setp(ax.get_xticklabels(), rotation=25) for ax in p.axes.flat]
    p.set_titles(row_template='{row_name}', col_template="{col_name}",fontweight='bold', size=18)
    p.fig.text(x = 0, y = 0.65, s='Area Under Curve', rotation = 90, size=18) # y axis label
    p.savefig(figurename)
    print 'Plotted %s' %figurename

def main():
    enzymes = ['tei','gsi']
    figurename = 'auc_plot.png'
    table_name = 'auc_plot.tsv'
    aucDF = map(readFile,enzymes)
    aucDF = pd.concat(aucDF,axis=0)
    aucDF.to_csv(table_name,sep='\t',index=False)
    print 'Written %s' %table_name
    aucDF = pd.read_csv(table_name,sep='\t')
    plot(aucDF, figurename)
    return 0

if __name__ == '__main__':
    main()
