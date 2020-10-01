import numpy as np

class principal_components:
    
    def __init__(self,data):
        self.data=data
    #returns eigen values
    def pca(self):
        c=np.cov(self.data.T)
        eigen_values,eigen_vectors=np.linalg.eig(c)
        #sort from highest to lowest
        idx=np.argsort(eigen_values)[::-1]
        eigen_values=eigen_values[idx]
        eigen_vectors=eigen_vectors[:,idx]
        return eigen_vectors,eigen_values

    #projects data to new subspace    
    def projection(self,data):
        p,e=self.pca()
        #projection
        x=data-np.mean(self.data,axis=0)
        return x@p

    def d_principal_components(self,data,d):
        w=self.projection(data)
        w=w[:,:d]
        return w

    def d_rank_approximation(self,data,d): 
        mu=np.mean(self.data,axis=0)
        w=self.projection(data)
        p,e=self.pca(self.data)
        w=w[:,:d]
        p=p[:,:d]
        x_hat=np.dot(w,p.T)+mu    
        return x_hat

    def variation(self,delta):
        p,e=self.pca()
        d=np.cumsum(e)/np.sum(e)

        idx=np.where(d>delta)

        return np.min(idx)

    def variance_contributions(self):
        p,e=self.pca()
        v=e/np.sum(e)
        return v
