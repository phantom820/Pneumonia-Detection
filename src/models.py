import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

class logistic_regression:
    def __init__(self,data,labels):
        self.x_train=data
        self.y_train=labels
        self.theta=np.random.randn(len(data[0]),)
        N=len(data)
        
    def sigmoid(self,x,theta):
        y=1/(1+np.exp(-np.dot(x,theta)))
        return y

    def cost_function(self,theta,gamma=0):
        x=self.x_train
        y=self.y_train
        h=self.sigmoid
        if theta.ndim>1:
            entropy=-y[:,np.newaxis]*np.log(h(x,theta.T)+10**-5)-(1-y[:,np.newaxis])*np.log(1-h(x,theta.T)+10**-5)
            J=np.mean(entropy,axis=0)
            return J
        else:
            entropy=-y*np.log(h(x,theta)+10**-5)-(1-y)*np.log(1-h(x,theta)+10**-5)+0.5*gamma*np.dot(theta,theta)
            J=np.mean(entropy,axis=0)
            
            return J
        
    def stochastic_gradient_descent(self,theta,alpha,epsilon,gamma,t_max=100):
        J=[]
        t=0
       
        while t<t_max:
          
            
            for i in range(len(self.x_train)):
                x=self.x_train
                y=self.y_train
                
                idx=np.arange(len(x))
                np.random.shuffle(idx)
                x=x[idx]
                y=y[idx]
                
                h=self.sigmoid        
                J.append(self.cost_function(theta,gamma=gamma))
                theta_prior=theta.copy()
                theta[0]=theta[0]-alpha*((h(x[i],theta)-y[i])*x[i][0])
                theta[1:]=theta[1:]*(1-alpha*gamma)-alpha*((h(x[i],theta)-y[i])*x[i][1:])

                residual=abs(theta-theta_prior)
                
                if np.linalg.norm(residual,ord=1)<epsilon or t>t_max:
                    return theta,J
            
                t=t+1
            
        return theta,J
            
    def mini_batch_gradient_descent(self,theta,alpha,epsilon,gamma,t_max,batch_size):
        J=[]
        t=0
        N=len(self.x_train)
        while t<t_max:
            x=self.x_train
            y=self.y_train
            for i in range(0,len(self.x_train),batch_size):
                idx=np.arange(len(self.x_train))
                np.random.shuffle(idx)    
                x=x[idx]
                y=y[idx]
                
                mini_batch=x[i:i+batch_size,:]
                mini_label=y[i:i+batch_size]
                h=self.sigmoid
                error=h(mini_batch,theta)-mini_label
                x_error=mini_batch*error[:,np.newaxis]

                J.append(self.cost_function(theta,gamma))

                grad=np.mean(x_error,axis=0)
                theta_prior=theta.copy()
                theta[0]=theta[0]-alpha*grad[0]
                theta[1:]=theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
                
                residual=theta-theta_prior
            
                if np.linalg.norm(residual)<epsilon:
                    return theta,J
                t=t+1
        return theta,J
         
        
    #bacth gradient by default uses full
    def batch_gradient_descent(self,theta,alpha,epsilon,gamma,t_max=1500):  
        J=[]
        t=0
        N=len(self.x_train)
        while t<t_max:
            h=self.sigmoid
            error=h(self.x_train,theta)-self.y_train
            x_error=self.x_train*error[:,np.newaxis]

            J.append(self.cost_function(theta,gamma))

            grad=np.mean(x_error,axis=0)
            
            theta_prior=theta.copy()
            theta[0]=theta[0]-alpha*grad[0]
            theta[1:]=theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
                
            residual=theta-theta_prior

            if np.linalg.norm(residual)<epsilon:
                return theta,J
            t=t+1
        return theta,J
    
    def genetic_algorithm(self,n,N=50,m=10,t_max=1000):

        f=self.cost_function
        population=np.zeros((N,n+1))
        population[:,:n]=np.random.randn(N,n)
        population[:,n]=f(population[:,:n])
        children=np.zeros((m,n+1))
        J=np.zeros((t_max))
        for t in range(t_max):
            population=population[np.argsort(population[:,-1])]
            J[t]=f(population[0,:n])
            l=np.arange(n)

            for k in range(0,m,2):

                for j in range(2):
                    n1,n2=1,1

                    while n1==n2:
                        n1=int(np.floor(np.random.uniform(0,1)*(N)))
                        n2=int(np.floor(np.random.uniform(0,1)*(N)))

                    if population[n1,n]<population[n2,n]:
                        l[j]=n1
                    else:
                        l[j]=n2

                x=np.zeros((n))
                y=np.zeros((n))
                for j in range(n):
                    alpha1=(-1/2)+((3/2)+1/2)*np.random.uniform(0,1)
                    x[j]=alpha1*population[l[0],j]+(1-alpha1)*population[l[1],j]
                    alpha2=(-1/2)+((3/2)+1/2)*np.random.uniform(0,1)
                    y[j]=alpha2*population[l[1],j]+(1-alpha2)*population[l[0],j]

                children[k,:n]=x
                children[k+1,:n]=y
                children[:,n]=f(children[:,:n])

            population[-m:,:]=children
        population=population[np.argsort(population[:,-1])]
        return population[0,:-1],J

#     def particle_swarm_optimization(self,n,N,l,u,w,c1,c2,f,t_max):

#         population=np.zeros((N,n+1))
#         velocity=np.zeros((N,n))
#         J=np.zeros((t_max,))
        
#         #initialize population
#         population[:,:n]=l+np.random.uniform(0,1,(N,n))*(u-l)
#         velocity[:,:]=0.2*np.random.uniform(0,1,(N,n))*(u-l)
#         population[:,n]=f(population[:,:n])

#         #copy the population
#         populationCopy=population.copy()

#         #calculate the best point in population copy just sort and select first
#         g=populationCopy[np.argsort(populationCopy[:,-1])][0,:n]

#         t=0

#         while t<t_max:
    
#             J[t]=f(g)
#             r1,r2=np.random.uniform(0,1,(N,n)),np.random.uniform(0,1,(N,n))
#             velocity=w*velocity+c1*r1*(populationCopy[:,:n]-
#                             population[:,:n])+c2*r2*(g-population[:,:n])

#             population[:,:n]=population[:,:n]+velocity

#             #maintain feasibility
#             ida=np.where((population[:,:n]<l))
#             idb=np.where((population[:,:n]>u))

#             if np.count_nonzero(ida==True)>0:
#                 population[ida]=l+np.random.uniform(0,1)*(u-l)

#             if np.count_nonzero(idb==True)>0:
#                 population[idb]=l+np.random.uniform(0,1)*(u-l)
                
#             #evaluate new solutions
#             population[:,n]=f(population[:,:n])

#             #update the matrix
#             idx=population[:,n]<populationCopy[:,n]
#             populationCopy[idx]=population[idx]

#             #calculate the best point in population copy just sort and select first
#             g=populationCopy[np.argsort(populationCopy[:,-1])][0,:n]

#             t=t+1
            
#         return g,J

   
                
    def predict(self,x,theta):
        h=self.sigmoid
        y_predict=h(x,theta)
        y_predict[y_predict>0.5]=1
        y_predict[y_predict<=0.5]=0
        return y_predict
    
    def train_model(self,mode,alpha=0.1,epsilon=10**-3,gamma=0.01,batch_size=128,
                    t_max=1000,N=50,m=10):
        if mode=="bgd":
            #print("Batch gradient descent learning")
            self.theta=np.random.randn(len(self.x_train[0]),)
            self.theta,self.J=self.batch_gradient_descent(self.theta,alpha=alpha,epsilon=epsilon,
                                                          gamma=gamma,t_max=t_max)
        
        elif mode=="mbgd":
            #print("Mini batch gradient descent learning")
            self.theta=np.random.randn(len(self.x_train[0]),)
            self.theta,self.J=self.mini_batch_gradient_descent(self.theta,alpha=alpha,epsilon=epsilon,
                                                               gamma=gamma,batch_size=batch_size,t_max=t_max)
        
        elif mode=="sgd":
            #print("Stochastic gradient descent learning")
            self.theta=np.random.randn(len(self.x_train[0]),)
            self.theta,self.J=self.stochastic_gradient_descent(self.theta,alpha=alpha,
                                                               epsilon=epsilon,gamma=gamma,t_max=t_max)
        elif mode=="ga":
           # print("Genetic algorithm learning")
            self.theta=np.random.randn(len(self.x_train[0]),)
            dimensions=len(self.x_train[0])
            self.theta,self.J=self.genetic_algorithm(n=dimensions,N=N,m=m,t_max=t_max)
        
#         elif mode=="pso":
#             print("particle swamp optimization")
#             self.theta=np.random.randn(len(self.x_train[0]),)
#             n=len(self.x_train[0])
#             l=np.array([lower for i in range(n)])
#             u=np.array([upper for i in range(n)])
#             f=self.cost_function
#             self.theta,self.J=self.particle_swarm_optimization(n,N,l,u,w,c1,c2,f,t_max)
    
    def training_fit(self):
        y_predict=self.predict(self.x_train,self.theta)
        return np.count_nonzero(y_predict==self.y_train)/len(self.y_train)
    
    def testing_fit(self,x_test,y_test):
        y_predict=self.predict(x_test,self.theta)
        return np.count_nonzero(y_predict==y_test)/len(y_test)  
        
    def confusion_matrix(self,x_test,y_test):
        y_predict=self.predict(x_test,self.theta)
        cm=confusion_matrix(y_test,y_predict)
#         print(np.count_nonzero((y_predict==y_test) & (y_test==0)))
#         print(np.count_nonzero((y_predict==y_test) & (y_test==1)))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Pneumonia', 'Normal']); ax.yaxis.set_ticklabels(['Normal', 'Pneumonia'])
        
        return ax   
        
import seaborn as sns

class kNN:
    def __init__(self, data, labels, k):
        self.x = np.hstack((data, labels[:, np.newaxis]))
        self.k = k
	# computes euclidean distance between a and b
    def distance(self, x, q):
        d = np.sum((x[:, :len(q)] - q) ** 2, axis=1)
        return np.round(np.sqrt(d), 2)

    # returns k nearest neighbours of query q along with distances
    def classifier(self, x, q, k):
        d = self.distance(x, q)
        idx = np.argsort(d)
        label = x[idx, -1].reshape(x.shape[0], 1)
        d = d[idx].reshape(x.shape[0], 1)
        return np.hstack((d, label))[:k, :]

    # predicts class of a point using kNN
    def predict(self, n):
        zero = np.count_nonzero(n[:, 1] == 0)
        one = np.count_nonzero(n[:, 1] == 1)
        if zero > one:
            return 0
        else:
            return 1

    def assign(self,x_test,y):
        x = np.hstack((x_test, y[:, np.newaxis]))
        correct = 0
        y_pred=np.zeros((len(y)))
        i=0
        for row in x:
            q = row[:-1]
            n = self.classifier(self.x, q, self.k)
            y_pred[i]=self.predict(n)
            i=i+1
            
        return y_pred
        
    def training_fit(self):
        correct = 0
        for row in self.x:
            q = row[:-1]
            c = row[-1]
            n = self.classifier(self.x, q, self.k)
            if c == self.predict(n):
                correct = correct + 1
        return correct / len(self.x)

    def testing_fit(self, x, y):
        x = np.hstack((x, y[:, np.newaxis]))
        correct = 0
        for row in x:
            q = row[:-1]
            c = row[-1]
            n = self.classifier(self.x, q, self.k)
            if c == self.predict(n):
                correct = correct + 1
        return correct / len(x)
        
        
    def confusion_matrix(self,x_test,y_test):
        y_predict=self.assign(x_test,y_test)
        cm=confusion_matrix(y_test,y_predict)
#         print(np.count_nonzero((y_predict==y_test) & (y_test==0)))
#         print(np.count_nonzero((y_predict==y_test) & (y_test==1)))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Pneumonia', 'Normal']); ax.yaxis.set_ticklabels(['Normal', 'Pneumonia'])
        
        return ax 
                
		           
class multilayer_perceptron:
    
    def __init__(self,x,y,architecture,activation="logistic",solver="adam",t_max=500,learning_rate=0.001):
        self.x_train=x
        self.y_train=y
        self.architecture=architecture
        self.network=MLPClassifier(activation=activation,max_iter=t_max,solver=solver,learning_rate_init=learning_rate,
                                   alpha=1e-5,hidden_layer_sizes=architecture, random_state=1)
        
    def train_model(self):
        self.network.fit(self.x_train,self.y_train)
    
    def training_fit(self):
        y_predict=(self.network.predict(self.x_train))
        return np.count_nonzero(self.y_train==y_predict)/len(self.y_train)
    
    def testing_fit(self,x_test,y_test):
        y_predict=(self.network.predict(x_test))
        return np.count_nonzero(y_test==y_predict)/len(y_test)
        
    def training_fit(self):
        y_predict=self.network.predict(self.x_train)
        return np.count_nonzero(y_predict==self.y_train)/len(self.y_train)
    
    def testing_fit(self,x_test,y_test):
        y_predict=self.network.predict(x_test)
        return np.count_nonzero(y_predict==y_test)/len(y_test)  
        
    def confusion_matrix(self,x_test,y_test):
        y_predict=(self.network.predict(x_test))
        cm=confusion_matrix(y_test,y_predict)
#         print(np.count_nonzero((y_predict==y_test) & (y_test==0)))
#         print(np.count_nonzero((y_predict==y_test) & (y_test==1)))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Pneumonia', 'Normal']); ax.yaxis.set_ticklabels(['Normal', 'Pneumonia'])
        
        return ax 
