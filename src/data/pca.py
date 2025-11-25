from sklearn.decomposition import PCA

def library_pca(data):
    #TODO
    pca = PCA(n_components=2) #just going with 2 for now
    pca.fit(data)
    return pca

def pca_by_hand():
    #TODO
    pass