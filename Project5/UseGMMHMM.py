import numpy as np
class GMMHMM(object):
    #please input all the templates for one specific word
    def __init__(self,templates,Gaussian_distribution_number=[4,4,4,4,4]):
        self.templates=templates
        #len should be state_number
        self.Gaussian_distribution_number=Gaussian_distribution_number
        self.state_number=len(self.Gaussian_distribution_number)
        self.node_in_each_state=[]
        self.node_state=[]
        self.hmm =None
class hmmInfo:
    '''hmm model param'''
    def __init__(self):
        self.init = [] #初始矩阵
        self.transition_cost = []
        self.mix = [] #高斯混合模型参数,有几个state，里面就有几个mix
        self.N = 0 #状态数
class mixInfo:
    """docstring for mixInfo"""
    def __init__(self):
        self.Gaussian_mean = []#每个gaussian distribution的 mean vector
        self.Gaussian_var = [] #每个gaussian distribution的 diagnol covarience
        self.Gaussian_weight = []#每个gaussian distribution 的权重（其和为1）
        self.Num_of_Gaussian = 0 #几个gaussian distribution
def getMFCC2(wavename):#without normalization
    import numpy as np
    import scipy.io.wavfile as wav
    from python_speech_features import mfcc
    fs, audio = wav.read(wavename)
    feature_mfcc = mfcc(audio, samplerate=fs)
    mfcc=[]
    mfcc.append(np.hstack([feature_mfcc[0],feature_mfcc[0],feature_mfcc[0]]))
    for i in range(1,len(feature_mfcc)-1):
        delta=np.zeros(13)
        for j in range(13):
            delta[j]=feature_mfcc[i+1][j]-feature_mfcc[i-1][j]
        mfcc.append(np.hstack([feature_mfcc[i],delta]))
    mfcc.append(np.hstack([feature_mfcc[-1],feature_mfcc[-1],feature_mfcc[-1]]))

    for i in range(1,len(mfcc)-1):
        acc=np.zeros(13)
        for j in range(13):
            acc[j]=mfcc[i+1][13+j]-mfcc[i-1][13+j]
        mfcc[i]=np.hstack([mfcc[i],acc])
    mfccs=np.array(mfcc)
    std=np.std(mfccs)
    var=np.var(mfccs,1)
    for i in range(len(mfccs)):
        for j in range(39):
            mfccs[i][j]=mfccs[i][j]/var[i]
    return mfccs
def log_gaussian(mu,squared_sigma,input_vector):
    #Author: Huangrui Chu, 
    #Calculate the cost using log gaussian
    part1=0.5*np.sum(np.log((2*np.pi)*(squared_sigma)),axis=1)
    part2=0.5*np.sum(np.square((input_vector-mu))/squared_sigma,axis=1)
    cost= part1+part2
    return cost

def gaussian(mu,squared_sigma,input_vector):
    #Author: Huangrui Chu, 
    #Calculate the probability, we only return a number!!!!
    #为了方便 我这边的一个numpy 推广就先应用到mu 上吧 毕竟我后面是一帧一帧的去分析
    #print(type(squared_sigma))
    #d=input_vector.shape[0]
    d=2
    part0=np.prod(squared_sigma,axis=1)# Huangrui use this to give the product of squared_sigma 1,2,...39
    part1=np.sqrt((2*np.pi)**d *part0)
    front=1/part1
    part2=0.5*np.sum((mu-input_vector)**2/squared_sigma,axis=1)
    expo=np.exp(-part2)
    p=front*expo
    #p=np.exp(log_gaussian(mu,squared_sigma,x))
    return p

def mixture_log_gaussian(mix,input_vector):
    weight=mix.Gaussian_weight
    mu = mix.Gaussian_mean
    squared_sigma = mix.Gaussian_var
    cost=log_gaussian(mu,squared_sigma,input_vector)
#     print(cost)
#     print(weight)
    weighted_cost=np.sum(weight*cost)
    return weighted_cost