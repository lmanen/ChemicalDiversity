#!/usr/bin/python3.5 -u
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import glob, os, re,csv,fnmatch, sys
import csv
import operator
from time import sleep
from time import *             #meaning from time import EVERYTHING
import time
import operator


import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import metrics,datasets
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, pairwise_distances_argmin_min
from sklearn.cluster import KMeans, SpectralClustering,AffinityPropagation
from sklearn.manifold import TSNE
from pyclust import KMedoids

import chart_studio.tools as tls
tls.set_credentials_file(username='pralins', api_key='4XIxgyLHkDsyXZIJrYEV')

import chart_studio.plotly as py
from plotly.graph_objects import * 
import plotly.graph_objs as go
import plotly.offline as ply

from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist,cdist

from itertools import cycle

import requests

from chemspipy import ChemSpider 
from more_itertools import locate


###PRINCIPAL COMPONENT ANALYSIS###
def pca(df,var, **kwargs):
    ''' 
    df = dataframe containing 1st columns of smiles and subsequent calculated molecular descriptors
    var = if int., the number of PCs will be fix / if 0<n<1 select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified

    The function will return the dataframe matrix containing the calculated PCA (columns) per N molecules (rows)

    if plot = True (default = None), it will return the interactive plot of the accumulated explained variance per PC.

    if variance = True (default = None), it will return the array of the explained variance per PC.
    '''
    #Reading the data
    X=df
    X = df.iloc[:,1:].values
    X_std=StandardScaler().fit_transform(X)

    #Para que te haga tantas PCAs necesarias para obtener varianza de 0.95
    acp = PCA(n_components=var)
    Y = acp.fit_transform(X_std)
    np.cumsum(acp.explained_variance_ratio_)
    number_PC=len(np.cumsum(acp.explained_variance_ratio_))
    variance=np.sum(acp.explained_variance_ratio_)

    #Hacemos las PCAs
    pca = PCA(whiten=True)
    PC=pca.fit_transform(X_std)
    PCs=pd.DataFrame(PC)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    cumsum

    #Graficamos la varianza acumulada
    plot1 = Bar(x=["PC%s"%i for i in range(1,20)], y = pca.explained_variance_ratio_, showlegend=False)
    plot2 = Scatter(x=["PC%s"%i for i in range(1,20)], y = cumsum, showlegend=True, name = "% Cumulative Variance")

    data = Data([plot1, plot2])

    layout = Layout(xaxis = XAxis(title="Principal Components"), 
                   yaxis = YAxis(title = "Variance percentage"),
                   title = "Percentage of variance per Principal Component")

    fig = Figure(data = data, layout = layout)
    print('The number of PCs needed is %s (with %s variance)' %(number_PC,variance))
#     return(pd.DataFrame(PCs.iloc[:,:number_PC]))    
    plot=kwargs.pop("plot", False) 
    if plot==True:
        return(py.iplot(fig)) 
    pca_var=kwargs.pop("variance", False) 
    if pca_var==True:
        return(np.array(pca.explained_variance_ratio_)[:number_PC])
    return(pd.DataFrame(PCs.iloc[:,:number_PC]))    


###DEFINING PARTITIONING ALGORITHMS###
#Functions needed for further partitioning purposes
def midrange(X, column):
    '''
    X = Dataframe containing PC values
    column = column specified to calculate its midrange
    Results in the calculation of the medium value of the PC
    '''
    mr= (max(X.iloc[:,column])+min(X.iloc[:,column]))/2
    return(mr)
def interval(X,col):
    '''
    X = Dataframe containing PC values
    column = column specified to calculate its interval
    Results in the calculation of the PC interval 
    '''
    ival=X.iloc[:,col].max()-X.iloc[:,col].min()
    return(ival)

#Binning algorithm#
def binning (X, NCLUST):
    '''
    For a given dataframe containing PCs (X) and desired maximum of bins to be calculated, it returns the array of the indexed molecules contained in each occupied bin. 
    '''
    df=X.copy()
    df['class']=np.NaN
    cut_value=df.iloc[:,0].min()+(interval(df,0)/2)
    for j,i in enumerate(df[df.columns[0]]):
        if i<cut_value:
            df['class'][j]=0
        else:
            df['class'][j]=1 
    for h in range(1,len(df.columns)-1):# avoids counting class column
        if len(np.unique(df['class']))<NCLUST:
            B=df.copy()
            print('dividing {} column'.format(h))
            cut_value=midrange(df,h) 
            for j,i in zip(df[h].index,df[h]):
                if i>=cut_value:
                     df['class'][j]=df['class'][j]+2**(h) #Se inicia en 1 porque ya se ha hecho la primera partición
            if len(np.unique(df['class']))>NCLUST:
                df=B.copy()
                print('{} Bins have been created, {} are occupied'.format(2**(h),len(np.unique(df['class']))))
                break
            else:
                print('{} Bins have been created, {} are occupied'.format(2**(h+1),len(np.unique(df['class']))))

    return(df)

#OV_binning algorithm#
def OV_binning (X, NCLUST, var):
    '''
    df = dataframe containing PCs
    NCLUST =  number of clusters desired (maximum to be binned)
    var = array of the explained variance per PC (obtained through pca (df, var, variance=True)
    The function will return an array with the bin index of each sorted molecule of the dataframe
    '''
    df=X.copy()
    var2=var.copy()
    sequence=[]
    print("PCA variances: {} \n".format(var2))
    df['class']=0

    ## It is firstly stablished the order of PCs to be divided sorting them by descending variance
    print("Stablishing PCA priority to be binned depending on its variance:\n")
    for z in range(len(var)):
        print("{} ROUND \n".format(z))
        index=np.where(var2[:z+1]==max(var2[:z+1]))#coge a partir de 1 pq cero es el valor minimo
        var2[index]=var2[index]/2
        sequence.extend(index[0])
        cut_number=var[index]/var2[index]
        print("{} partitions: {} ".format(z+1, sequence))
        print(var2)
    ## Considering the priorization, the indexes per bin forned are assigned
    for z,v in enumerate(sequence):
        if len(np.unique(df['class']))<NCLUST:
            B=df.copy()
            cut_number=2**(sequence[:z+1].count(v))
            print("{} cuts are made in {} PCA\n".format(cut_number, v))# It considers +1 because it does not take in acconunt a final cutting number, it will be calculated 2^n  bins being n the number of times where the PC has been divided
            steps=int(cut_number/2)
            range_values=[]
            for i in range(int(steps)):
                range_values.append(list(np.linspace(min(df[v]),max(df[v]),int(cut_number)+1,endpoint=True)[(i*2+1):((i+1)*2+1)])) # It picks the upper quartiles to stablish the upper limits 
            for i,j in enumerate(range_values):
                print(j)
                for d,f in zip(df[v].index,df[v]):
                    if f > j[0] and f <= j[1]:
                        df['class'][d]=df['class'][d]+2**(z)
            if len(np.unique(df['class']))>NCLUST:
                df=B.copy()
                print('{} Bins have been created, {} are occupied\n'.format(2**(z),len(np.unique(df['class']))))
                break
            else:
                print('{} Bins have been created, {} are occupied\n'.format(2**(z+1),len(np.unique(df['class']))))     
    return(df)

#CLUSTERING#
def Cluster(df, X, NCLUST, clustering_names, **kwargs):
    '''
    df = original dataframe containing SMILES and molecular descriptors (needed for OV_BIN)
    X = dataframe containing PCs
    NCLUST =  number of clusters desired (maximum to be binned)
    clustering_names: an array of clustering methods from the possible available: 'HRC_single', 'HRC_complete', 'HRC_median', 'HRC_average', 'HRC_centroid', 'HRC_ward', 'KMN', 'KMED', 'spectral_clust', 'aff_prop', 'BINNING', 'OV_BIN' the functions retrieves the X_Cluster matrix containing the matrix of PCA followed by the clustering columns with the molecule index. 
    
    clustering_names = 'All' is available to calculate all the clusterings available if possible
    '''
    if clustering_names=='All':
        clustering=[]
        clustering_names_array=[]
        HRC_names=[]
        #HRC single
        try:
            print('Processing HRC single clustering \n')
            HRC_single=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='single')
            clustering.append(HRC_single)
            clustering_names_array.append("HRC_single")
            HRC_names.append("HRC_single")
        except:
            print('Could not calculate HRC single clustering \n')

        #HRC complete
        try:
            print('Processing HRC complete clustering\n')
            HRC_complete=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='complete')
            clustering.append(HRC_complete)
            clustering_names_array.append("HRC_complete")
            HRC_names.append("HRC_complete")
        except:
            print('Could not calculate HRC complete clustering\n')

        #HRC median
        try:
            print('Processing HRC median clustering\n')
            HRC_median=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='median')
            clustering.append(HRC_median)
            clustering_names_array.append("HRC_median")
            HRC_names.append("HRC_median")
        except:
            print('Could not calculate HRC median clustering\n')

        #HRC average    
        try:
            print('Processing HRC average clustering\n')
            HRC_average=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='average')
            clustering.append(HRC_average)
            clustering_names_array.append("HRC_average")
            HRC_names.append("HRC_average")
        except:
            print('Could not calculate HRC average clustering\n')

        #HRC centroid
        try:
            print('Processing HRC centroid clustering\n')
            HRC_centroid=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='centroid')
            clustering.append(HRC_centroid)
            clustering_names_array.append("HRC_centroid")
            HRC_names.append("HRC_centroid")
        except:
            print('Could not calculate HRC centroid clustering\n')

        #HRC ward
        try:
            print('Processing HRC ward clustering\n')
            HRC_ward=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='ward')
            clustering.append(HRC_ward)
            clustering_names_array.append("HRC_ward")
            HRC_names.append("HRC_ward")
        except:
            print('Could not calculate HRC ward clustering\n')


        #KMN: K-means
        try:
            print('Processing KMN clustering\n')
            KMN = KMeans(NCLUST).fit_predict(X)
            clustering.append(KMN)
            clustering_names_array.append("KMN")
        except:
            print('Could not calculate KMN clustering\n')


        #KMED: K-medoid
        try:
            print('Processing KMED clustering\n')
            data=np.array(X)
            KMED = KMedoids(NCLUST).fit_predict(data)
            clustering.append(KMED)
            clustering_names_array.append("KMED")
        except:
            print('Could not calculate KMED clustering\n')


        #Spectral clustering
        try:
            print('Processing Spectral clustering\n')
            spectral_clust = SpectralClustering(NCLUST).fit_predict(data)
            clustering.append(spectral_clust)
            clustering_names_array.append("spectral_clust")
        except:
            print('Could not calculate spectral clustering\n')

        # Binning (no variance priority)
        try:
            print('Processing  Binning clustering\n')
            BIN=binning(X,NCLUST)
            BIN=np.array(BIN.iloc[:,-1])
            BIN=BIN.astype(int)
            for i,j in enumerate(np.unique(BIN)):
                BIN[np.where(BIN==j)]=i
            clustering.append(BIN.astype(int))
            clustering_names_array.append("BINNING")
        except:
            print('Could not calculate  Binning clustering\n')

        #Optimum Variance Binning (PCA subdivision attending to variance priority)
        try:
            print('Processing Optimum Variance Binning clustering\n')
            variance=pca(df,0.95,variance=True)
            OV_BIN=OV_binning(X,NCLUST,variance)
            OV_BIN=np.array(OV_BIN.iloc[:,-1])
            OV_BIN=OV_BIN.astype(int)
            for i,j in enumerate(np.unique(OV_BIN)):
                OV_BIN[np.where(OV_BIN==j)]=i
            clustering.append(OV_BIN.astype(int))
            clustering_names_array.append("OV_BIN")
        except:
            print('Could not calculate Optimum Variance Binning clustering\n')

        print('Clustering calculation done!\n')

        X_cluster=X.copy()
        for i in range(len(clustering)):
            X_cluster[clustering_names_array[i]]=clustering[i]

        for i in HRC_names: #As HRC methods start enumeration in 1 whereas other methods start in 0, a standarization is needed. 

            try:
                X_cluster[i]=X_cluster[i]-1
            except:
                continue
        return(X_cluster)
    else:
        clustering=[]
        clustering_names_array=[]
        HRC_names=[]
        for i in clustering_names:
            if i == 'HRC_single':
                #HRC single
                try:
                    print('Processing HRC single clustering \n')
                    HRC_single=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='single')
                    clustering.append(HRC_single)
                    clustering_names_array.append("HRC_single")
                    HRC_names.append("HRC_single")
                except:
                    print('Could not calculate HRC single clustering \n')

            if i == 'HRC_complete':
            #HRC complete
                try:
                    print('Processing HRC complete clustering\n')
                    HRC_complete=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='complete')
                    clustering.append(HRC_complete)
                    clustering_names_array.append("HRC_complete")
                    HRC_names.append("HRC_complete")
                except:
                    print('Could not calculate HRC complete clustering\n')

            if i=='HRC_median':
                #HRC median
                try:
                    print('Processing HRC median clustering\n')
                    HRC_median=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='median')
                    clustering.append(HRC_median)
                    clustering_names_array.append("HRC_median")
                    HRC_names.append("HRC_median")
                except:
                    print('Could not calculate HRC median clustering\n')

            if i== 'HRC_average':
                #HRC average    
                try:
                    print('Processing HRC average clustering\n')
                    HRC_average=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='average')
                    clustering.append(HRC_average)
                    clustering_names_array.append("HRC_average")
                    HRC_names.append("HRC_average")
                except:
                    print('Could not calculate HRC average clustering\n')

            if i=='HRC_centroid':
                #HRC centroid
                try:
                    print('Processing HRC centroid clustering\n')
                    HRC_centroid=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='centroid')
                    clustering.append(HRC_centroid)
                    clustering_names_array.append("HRC_centroid")
                    HRC_names.append("HRC_centroid")
                except:
                    print('Could not calculate HRC centroid clustering\n')

            if i=='HRC_ward':
                #HRC ward
                try:
                    print('Processing HRC ward clustering\n')
                    HRC_ward=fclusterdata(X, criterion='maxclust', t=NCLUST, metric='euclidean', method='ward')
                    clustering.append(HRC_ward)
                    clustering_names_array.append("HRC_ward")
                    HRC_names.append("HRC_ward")
                except:
                    print('Could not calculate HRC ward clustering\n')

            if i=='KMN':
                #KMN: K-means
                try:
                    print('Processing KMN clustering\n')
                    KMN = KMeans(NCLUST).fit_predict(X)
                    clustering.append(KMN)
                    clustering_names_array.append("KMN")
                except:
                    print('Could not calculate KMN clustering\n')

            if i=='KMED':
                #KMED: K-medoid
                try:
                    print('Processing KMED clustering\n')
                    data=np.array(X)
                    KMED = KMedoids(NCLUST).fit_predict(data)
                    clustering.append(KMED)
                    clustering_names_array.append("KMED")
                except:
                    print('Could not calculate KMED clustering\n')

            if i=="spectral_clust":
                #Spectral clustering
                try:
                    print('Processing Spectral clustering\n')
                    spectral_clust = SpectralClustering(NCLUST).fit_predict(data)
                    clustering.append(spectral_clust)
                    clustering_names_array.append("spectral_clust")
                except:
                    print('Could not calculate spectral clustering\n')

            if i=='BINNING':
                # Binning (no variance priority)
                try:
                    print('Processing  Binning clustering\n')
                    BIN=binning(X,NCLUST)
                    BIN=np.array(BIN.iloc[:,-1])
                    BIN=BIN.astype(int)
                    for i,j in enumerate(np.unique(BIN)):
                        BIN[np.where(BIN==j)]=i
                    clustering.append(BIN.astype(int))
                    clustering_names_array.append("BINNING")
                except:
                    print('Could not calculate  Binning clustering\n')

            if i=='OV_BIN':
                #Optimum Variance Binning (PCA subdivision attending to variance priority)
                try:
                    print('Processing Optimum Variance Binning clustering\n')
                    variance=pca(df,0.95,variance=True)
                    OV_BIN=OV_binning(X,NCLUST,variance)
                    OV_BIN=np.array(OV_BIN.iloc[:,-1])
                    OV_BIN=OV_BIN.astype(int)
                    for i,j in enumerate(np.unique(OV_BIN)):
                        OV_BIN[np.where(OV_BIN==j)]=i
                    clustering.append(OV_BIN.astype(int))
                    clustering_names_array.append("OV_BIN")
                except:
                    print('Could not calculate Optimum Variance Binning clustering\n')

            print('Clustering calculation done!\n')

        X_cluster=X.copy()
        for i in range(len(clustering)):
            X_cluster[clustering_names_array[i]]=clustering[i]

        for i in HRC_names: #As HRC methods start enumeration in 1 whereas other methods start in 0, a standarization is needed. 
            try:
                X_cluster[i]=X_cluster[i]-1
            except:
                continue
        return(X_cluster)

#CLUSTERING ANALYSIS#
def PopDistribution(df2,clustering_name):
    '''
    df2 = with id of cluster ( typically X_cluster) for each compound
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    popdistribution=[]
    '''
    popdistribution=[]
    for i in range(max(df2[str(clustering_name)])+1):
        population=len(df2[df2[str(clustering_name)]==i])
        popdistribution.append(population)
    return(popdistribution)

def PopDistribution_Boxplots(df2,clustering_names,PNGname, fliers): 
    '''
    df2 = datafrsame with id of cluster ( typically X_cluster) for each compound
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    PNGname = stablish the name to save the boxplot as *.png
    fliers = True/False - If True, the clustering performance leading to overcrowded clusters can be identified
    The function retrieves the corresponding boxplots representing the population distribution depending on the aglorithm (clustering/partitioning methodology) performance
    ''' 
    Population_Matrix=pd.DataFrame()
    for i,j in zip(clustering_names,range(len(clustering_names))):    
        Population_Matrix=pd.concat([Population_Matrix,pd.DataFrame(PopDistribution(df2,i))], axis=1, join='outer') 

    Population_Matrix.columns=list(clustering_names)

    PM=[]
    for i in np.array(Population_Matrix.T):
        PM.append(i[~np.isnan(i)])       
        
    if fliers==True:
        fig1, ax1 = plt.subplots()
        ax1.set_title('Population Distribution')
        ax1.boxplot(PM, notch=True,labels=clustering_names)
        ax1.tick_params(axis='both', labelsize=10, labelrotation = 90)
        plt.savefig(('{}_popdistboxplot.png'.format(PNGname)),bbox_inches='tight', dpi=2000 ) 

    if fliers==False:        
        fig2, ax2 = plt.subplots()
        ax2.set_title('Population Distribution')
        ax2.boxplot(PM, showfliers=False, labels=clustering_names)
        ax2.tick_params(axis='both', labelsize=10, labelrotation = 90)
        plt.savefig(('{}_popdistboxplot_nofliers.png'.format(PNGname)),bbox_inches='tight', dpi=2000 )

def Cluster3d_distribution(df,cluster_name):
    '''
    df = datafrsame with id of cluster (typically X_cluster) for each compound
    cluster_name = clustering/partitioning method to be further analysed
    If a clustering method appears to have an unbalanced distribution an analysis can be made with this function deeper studying the number of analogs of the specified cluster. 
    The function will retrieve a plot bar of the population frequency of each cluster and information regarding the most and less populated clusters. It will further request which cluster is wanted to be explored and will show the chemmical space in a  3D plot with with the cluster's molecules colored in red
    '''
    list1=sorted(list(df[cluster_name]))
    nclust=max(list1)
#     clusterdist= {('cluster {}'.format(i)):list1.count(i) for i in list1}
    clusterdist= {i:list1.count(i) for i in list1}
    print('Cluster distribution: ')
    fig, ax1 = plt.subplots(figsize=(20, 4))
    fig.subplots_adjust(left=0.115, right=0.88)
    plt.bar(x=range(nclust+1), height=[clusterdist[i] for i in range(nclust+1)])#, tick_label=range(nclust))
    plt.show()
#     print(clusterdist)
    print('Most populated cluster: {}'.format(max(clusterdist, key=clusterdist.get)))
    print('Less populated cluster: {}'.format(min(clusterdist, key=clusterdist.get)))
    clusternumber=input('Which cluster would you like to visualize? ')
    clusternumber=int(clusternumber)

    PM=[]
    for i in df[cluster_name]:
        PM.append(i)  
    matrix=[]
    matrixindex=[i for i in range(len(PM)) if PM[i]==clusternumber]  
    for i in matrixindex:
         matrix.append(df.iloc[int(i),:3])
    matrix=pd.DataFrame(matrix)

    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs = df.iloc[:,0], ys = df.iloc[:,1], zs=df.iloc[:,2], c='LightGrey', marker= '.', alpha=0.05)
    ax.scatter(xs = matrix.iloc[:,0], ys = matrix.iloc[:,1], zs=matrix.iloc[:,2], c='Red', marker= 'o', alpha=1)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    
 ## Visualization plot
def Clusters_3dplot(X,X_cluster,clustering_names):
    '''
    X_cluster = dataframe with id of cluster for each compound
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    It returns different 3D plots where the analogs of the Chemical Space are colored depending on its cluster index
    '''
    for j in clustering_names:
        clusters=X_cluster[j]
        ##3D
        fig = plt.figure(figsize=(10,7.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs = X_cluster.iloc[:,0], ys = X_cluster.iloc[:,1], zs=X_cluster.iloc[:,2], c = clusters, cmap="hsv", marker= 'o', alpha=1)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        ax.set_title('{} Selection'.format(j))
 
 #BIBLIOGRAPHIC DATA SEARCHER IN PUBCHEM
def PubChemBibliographicDataSearcher(smiles2):
    '''
    smiles = Array of smiles (typically df['mol'])
    It will return an array of the library index of coincident reported molecules and their corresponding PubChem CIDs
    '''
    t0=time.time() 
    prolog = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
    pubchem_index=[]
    pubchem_CIDs=[]
    for j,i in enumerate(smiles2):
        can_smile=smiles2[j].replace("#","%23").replace("/","")
        url = prolog + "/compound/smiles/cids/txt?" + "smiles=" + can_smile
        res = requests.get(url)      
        if not int(res.text)==0:
            pubchem_CIDs.append(int(res.text))
            pubchem_index.append(j)
        if ( j % 5 == 4 ) :  # the % is the modulo operator and returns the remainder of a calculation (if i = 4, 9, ...)
            time.sleep(1) #Warning: When you make a lot of programmatic access requests using a loop, you should limit your request rate to or below five requests per second. Violation of usage policies may result in the user being temporarily blocked from accessing PubChem (or NCBI) resources** 
    print("Time of execution [min]:{}".format((time.time()-t0)/60))
    return(pd.DataFrame({'index':pubchem_index, 'CID':pubchem_CIDs}))

#RATIONAL SELECTIONS

class Selection:
    def __init__(self, X, df2):
        '''
        X = the matrix containing PCAs 
        df2 = the matrix df + the columns containing the clustering id (typically X_cluster)
        '''
        self.len_pca=len(X.columns)
        self.X = X
        self.df2 = df2 
    def centroid_selection(self,clustering): 
    #Returns an array of centroid compounds
        df2=self.df2
        X=self.X
#         clustering= input('Choose the clustering method for centroid selection {}:'.format(str(df2.columns.values[self.len_pca:])))          
        df3=df2.groupby([str(clustering)]).mean().iloc[:,0:self.len_pca]
        df3.columns=list(X.columns.values)
        selection=[]
        distances=[]
        for i in range(len(df3)):
            df4=df2[df2[str(clustering)]==i]
            df4=df4.iloc[:,0:self.len_pca]
            df5=df3.iloc[i,:]
            df5=pd.DataFrame(df5).T
            selection_raw, distances_raw= pairwise_distances_argmin_min(df5, df4)
            selection2=df4.index.values[int(selection_raw)]
            selection.append(selection2)
            distances.append(float(distances_raw))
        return(selection)

    def cherrypicking_selection(self, clustering):
        df2=self.df2
        X=self.X
        selection=[]
        for i in range(max(df2[str(clustering)])+1):
            df4=df2[df2[str(clustering)]==i]
            selection2=df4.sample(1, axis=0) #fix seed with random_state= ?
            selection2=selection2.index.values[0]
            selection.append(selection2)
        return(selection)
    
    def directed_selection_results(self, clustering, filters): 
        '''
        filters = it is specified the array of synthetically feasible molecules
        clustering = clustering method of interest
        Given an array of the accesible molecules (in the example, 10 randomly picked molecules will be considered), 
        this function will return a matrix with the sorted the molecules within the cluster from ascending distance to cluster centroid 
        '''
        #returns a selection of compounds preferebly with Lipinski=0 restriction and nearest to cluster's centroid
        df2=self.df2
        X=self.X
        len_pca=self.len_pca
        df4=pd.DataFrame([df2.loc[i] for i in filters])
        df3=df2.groupby([str(clustering)]).mean().iloc[:,0:len_pca]
        df3.columns=list(X.columns.values)
        selection_matrix=pd.DataFrame()
        for i in np.array(np.unique(df4[clustering])):
            df6=df4[df4[str(clustering)]==int(i)]
            df6=df6.iloc[:,0:len_pca]
            df5=pd.DataFrame(df3.iloc[int(i),:]).T
            A=sklearn.metrics.pairwise.euclidean_distances(df5, df6)
            df6['distance']=A[0]
            df6[clustering]=i
            df6=df6.sort_values(by=['distance'])
            selection_matrix=pd.concat([selection_matrix,df6])
        return(selection_matrix)

#Check COMMERCIAL AVAILABILITY OF REAGENTS
class ReagentSearcher:
    '''
    vendors = list of trusted vendors from 
    api_key = must be defined previously via https://developer.rsc.org/
    '''
    def __init__(self, vendors, api_key):
        self.vendors=vendors
        self.api_key=api_key
        
    def _ReagentSearcher_(self, smi, method2):#returns False/True if unique smile is in 
        #1st step: Convert smile to iupac name (Chemspipy molecule searcher seems not to convert properly Smile to CSID )
        url4='http://cactus.nci.nih.gov/chemical/structure/'+smi+'/iupac_name'
        iupac_name=requests.get(url4)
        #2nd step: a list of CSID is created 
        CSID=[]
        cs=ChemSpider(self.api_key)
        for result in cs.search(iupac_name.text):
            CSID.append(result.record_id)
        suppliers=[]    
        if method2=='boolean':
            for i in range(len(CSID)):    
                return(bool(cs.get_external_references(str(CSID[i]), datasources=self.vendors)))  
        if method2=='vendors':
            for i in range(len(CSID)):    
                ref=cs.get_external_references(str(CSID[i]), datasources=self.vendors)
                suppliers.append([ref[i]['source'] for i in range(len(ref))])
            suppliers=np.unique(suppliers)
            return(suppliers)
        
    def suppliers(self,smiles_vector, method):
        if method=='boolean':
            suppliers=[self._ReagentSearcher_(smi=smiles_vector[i], method2=method)  for i in range(len(smiles_vector))]
            suppliers=list(locate(suppliers, lambda i: i == True))
        if method=='vendors':
            suppliers=[self._ReagentSearcher_(smi=smiles_vector[i], method2=method)  for i in range(len(smiles_vector))]
        return(suppliers)

# COVERAGES CALCULATIONS
def SpaceCoverage(vector, df2, clustering_name): 
    '''
    vector= ID of selected compounds, df2=matrix with classes, clustering_name=column
    df2 = dataframe with id of cluster for each compound (typically X_cluster)
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    It returns the SC value.
    '''
    occupied_clusters=[]
    for i in vector:
        try:
            cluster=df2[clustering_name][i]
            occupied_clusters.append(cluster)
        except:
            continue
    spacecoverage=100*(len(np.unique(occupied_clusters)))/(max(df2[clustering_name])+1) #+1 because it starts in 0
    return(spacecoverage)

def PopCoverage(vector, df2, clustering_name): 
    '''
    vector= ID of selected compounds, df2=matrix with classes, clustering_name=column
    df2 = dataframe with id of cluster for each compound (typically X_cluster)
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    It returns the PC value.
    '''
    occupied_clusters=[]
    pop_occupied_clusters=[]
    for i in vector:
        try:
            cluster=df2[clustering_name][i]
            occupied_clusters.append(cluster)
        except:
            continue
    for i in np.unique(occupied_clusters):
        population=len(df2[df2[str(clustering_name)]==i])
        pop_occupied_clusters.append(population)
    pop_coverage=100*sum(pop_occupied_clusters)/len(df2) #+1 because it starts in 0
    return(pop_coverage)

class coverage_heatmap:
    def __init__(self, df, df2, clustering_names, **kwargs):
        '''
        df= PCA dataframe (typically X)
        df2 = dataframe with id of cluster for each compound (typically X_cluster)
        clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
        '''  
        self.len_pca=len(df.columns)
        self.df = df
        self.df2 = df2
        self.clustering_names=clustering_names
        random_control=kwargs.pop("random_control", False)       
        self.random_control = random_control

        #kwargs: random_control,selection_method: centroid/cherrypicking, coverage: space/population, savePNG

    def spacecov(self, selection_method, **kwargs): 
        '''
        Two methods for cluster selection: Centroid/Cherry Picking
        If random_control = True, a random study of NCLUST randomly selected molecues will be showed as the mean value of 5,000 repetitions
        SpaceCov Heatmaps will be shown in red
        The heatmap can be saved as *.csv file if specifying CSVname=True and as *.png file if savePNG=True
        '''
        df=self.df
        df2=self.df2
        clustering_names=self.clustering_names      
        if selection_method=='Centroid':
            selection=[]
            for i in clustering_names:
                df2=self.df2
                sele=Selection(df,df2)
                selection=selection + [sele.centroid_selection(i)] #[pd.Series(sele.centroid_selection(i))]

        if selection_method=='Cherry Picking':
            selection=[]
            for i in clustering_names:
                df2=self.df2
                sele=Selection(df,df2)
                selection=selection+[sele.cherrypicking_selection(i)] #[pd.Series(sele.cherrypicking_selection(i))]    
        
        name=input('Enter name of the CSV file to save dataframe of selected compounds: ')
        pd.DataFrame(selection).to_csv('{}.csv'.format(name),index=False)
        
        spacecov=[]
        for i in selection:
            for j in clustering_names:
                spacecov.append(SpaceCoverage(i,df2,j))
                
        spacecov=np.split(np.array(spacecov),len(clustering_names))

        if self.random_control==True:
            rand_spacecov=[]
            for i in range(500):
                df2=self.df2
                k=max(df2.iloc[:,len(df.columns)+1])+1
                random_samp=df2.sample(k, axis=0)
                random_samp=random_samp.index.values
                for j in self.clustering_names:
                    rand_spacecov.append(SpaceCoverage(random_samp,df2,j))

            rand_spacecov=pd.DataFrame(np.split(np.array(rand_spacecov),len(self.clustering_names))).T #The mean value of all random repetitions is calculated
            clustering_names2=self.clustering_names+['RANDOM']    
            spacecov = pd.concat([pd.DataFrame(spacecov), pd.DataFrame(rand_spacecov.mean()).T], axis=0)
        else:
            clustering_names2=self.clustering_names
           
        color=sns.color_palette("Reds")
        sns.heatmap(spacecov, annot=True, cmap=color, xticklabels=clustering_names, yticklabels=clustering_names2, fmt='.1f')
        plt.title('Space Coverage', fontsize = 15)
        plt.ylabel('{} Selected Compounds'.format(selection_method), fontsize=10) 
        plt.xlabel('Cluster', fontsize=10) 
                  
        saveCSV=kwargs.pop("saveCSV", False) 
        if saveCSV==True:
            CSVname=input('Enter name of the CSV file to save the Space Coverage heatmap: ')
            pd.DataFrame(spacecov).to_csv('{}.csv'.format(CSVname),index=False)
            
        savePNG=kwargs.pop("savePNG", False) 
        if savePNG==True:
            PNGname=input('Enter name of the PNG file to save the Space Coverage heatmap: ')
            plt.savefig(('{}.png'.format(PNGname)),bbox_inches='tight', dpi=2000 )    


    def popcov(self, selection_method, **kwargs):         
        '''
        Two methods for cluster selection: Centroid/Cherry Picking
        If random_control = True, a random study of NCLUST randomly selected molecues will be showed as the mean value of 5,000 repetitions
        PopCov Heatmaps will be shown in green
        The heatmap can be saved as *.csv file if specifying CSVname=True and as *.png file if savePNG=True
        '''
        df=self.df
        df2=self.df2
        clustering_names=self.clustering_names                      
        if selection_method=='Centroid':
            selection=[]
            for i in clustering_names:
                df2=self.df2
                sele=Selection(df,df2)
                selection=selection + [sele.centroid_selection(i)] #[pd.Series(sele.centroid_selection(i))]
                
        if selection_method=='Cherry Picking':
            selection=[]
            for i in clustering_names:
                df2=self.df2
                sele=Selection(df,df2)
                selection=selection+[sele.cherrypicking_selection(i)] #[pd.Series(sele.cherrypicking_selection(i))]            

        name=input('Enter name of the CSV file to save dataframe of selected compounds: ')
        pd.DataFrame(selection).to_csv('{}.csv'.format(name),index=False)                
                
        popcov=[]
        for i in selection:
            for j in clustering_names:
                popcov.append(PopCoverage(i,df2,j))
                
        popcov=np.split(np.array(popcov),len(clustering_names))

        if self.random_control==True:
            rand_popcov=[]
            for i in range(500):
                df2=self.df2
                k=max(df2.iloc[:,len(df.columns)+1])+1
                random_samp=df2.sample(k, axis=0)
                random_samp=random_samp.index.values
                for j in self.clustering_names:
                    rand_popcov.append(PopCoverage(random_samp,df2,j))

            rand_popcov=pd.DataFrame(np.split(np.array(rand_popcov),len(self.clustering_names))).T #Se hace la media del space coverage de todas las elecciones random realizadas. 
            clustering_names2=self.clustering_names+['RANDOM']    
            popcov = pd.concat([pd.DataFrame(popcov), pd.DataFrame(rand_popcov.mean()).T], axis=0)
        else:
            clustering_names2=self.clustering_names
           
        color=sns.color_palette("Greens")
        sns.heatmap(popcov, annot=True, cmap=color, xticklabels=clustering_names, yticklabels=clustering_names2, fmt='.1f')
        plt.title('Population Coverage', fontsize = 15)
        plt.ylabel('{} Selected Compounds'.format(selection_method), fontsize=10) 
        plt.xlabel('Cluster', fontsize=10) 
                  
        saveCSV=kwargs.pop("saveCSV", False) 
        if saveCSV==True:
            CSVname=input('Enter name of the CSV file to save the Population Coverage heatmap: ')
            pd.DataFrame(popcov).to_csv('{}.csv'.format(CSVname),index=False)
            
        savePNG=kwargs.pop("savePNG", False) 
        if savePNG==True:
            PNGname=input('Enter name of the PNG file to save the Population Coverage heatmap: ')
            plt.savefig(('{}.png'.format(PNGname)),bbox_inches='tight', dpi=2000 ) 

#t-SNE visualization plot
def tsne_pylins (pca_df, perp, itt):
    '''
    pca_df= PCA dataframe (typically X)
    perp = perplexity value 
    itt = number of iterations (typically 5000)
    '''
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=itt, init='pca')
    tsne_results = tsne.fit_transform(pca_df)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    X=pd.DataFrame(pca_df)
    X['tsne-PCA-one'] = tsne_results[:,0]
    X['tsne-PCA-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    plt.title("perplexity_{}_itt_{}".format(perp,itt))
    fig=sns.scatterplot(
        x="tsne-PCA-one", y="tsne-PCA-two",
    #     hue="class",
    #     palette=sns.color_palette("hls", NCLUST),
        data=X,
        legend="full",
        alpha=0.3
    )
   
    fig2=fig.get_figure()

def Clusters_2dtsneplot(X, X_cluster2, clustering_names):
    '''
    X= PCA dataframe 
    X_cluster2 = clustering dataframce with tsne indexes ("tsne-pca-one","tsne-pca-two") in two of its  columns
    itt = number of iterations (
    clustering_names = array of names as characters of each clustering (X_Clusters clustering columns)
    '''    
    for i in clustering_names:
        plt.figure(figsize=(16,10))
        plt.title("{}".format(i))
        fig=sns.scatterplot(
            x="tsne-pca-one", y="tsne-pca-two",
            hue=i,
            palette=sns.color_palette("hls", max(X_cluster2[i])+1),
            data=X_cluster2,
            legend="full",
            alpha=1
        )
