from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#def calculateThreshold():

def trainNetwork(inp):

    print ("After training the network")

    net=buildNetwork(1,4,1)

    ds=SupervisedDataSet(1,1)

    ds.addSample((0.4524,),(0.1,))
    ds.addSample((0.4688,),(0.12,))
    ds.addSample((0.5156,),(0.145,))
    ds.addSample((0.4609,),(0.16,))
    ds.addSample((0.4531,),(0.125,))
    ds.addSample((0.5547,),(0.125,))
    ds.addSample((0.9922,),(0.235,))
    ds.addSample((0.9922,),(0.2578,))
    ds.addSample((0.8672,),(0.2031,))
    ds.addSample((0.9624,),(0.25,))

    '''
    ds.addSample((0.9417,),(0.1718,))
    ds.addSample((0.9216,),(0.21,))
    ds.addSample((0.8359,),(0.1893,))
    ds.addSample((0.8281,),(0.1762,))
    ds.addSample((0.8828,),(0.1350,))
    ds.addSample((0.9893,),(0.31,))
    ds.addSample((0.6016,),(0.1508,))
    ds.addSample((0.5781,),(0.11,))
    ds.addSample((0.5078,),(0.095,))
    ds.addSample((0.5156,),(0.12,))
    ds.addSample((0.8100,),(0.145,))
    ds.addSample((0.5150,),(0.12,))
    ds.addSample((0.5859,),(0.1,))
    ds.addSample((0.5781,),(0.102,))
    ds.addSample((0.6172,),(0.16,))
    ds.addSample((0.4424,),(0.101,))
    ds.addSample((0.4788,),(0.121,))
    ds.addSample((0.5056,),(0.1451,))
    ds.addSample((0.4709,),(0.162,))
    ds.addSample((0.4431,),(0.1251,))
    ds.addSample((0.5647,),(0.1253,))
    ds.addSample((0.9822,),(0.2351,))
    ds.addSample((0.9922,),(0.2576,))
    ds.addSample((0.8572,),(0.2038,))
    ds.addSample((0.9724,),(0.2503,))
    ds.addSample((0.9517,),(0.1714,))
    ds.addSample((0.9116,),(0.2103,))
    ds.addSample((0.8459,),(0.1897,))
    ds.addSample((0.8181,),(0.1768,))
    ds.addSample((0.8928,),(0.1349,))
    ds.addSample((0.9793,),(0.3103,))
    ds.addSample((0.6116,),(0.1502,))
    ds.addSample((0.5681,),(0.1102,))
    ds.addSample((0.5178,),(0.0951,))
    ds.addSample((0.5056,),(0.1202,))
    ds.addSample((0.8200,),(0.1451,))
    ds.addSample((0.5050,),(0.1204,))
    ds.addSample((0.5759,),(0.1,))
    ds.addSample((0.5881,),(0.1022,))
    ds.addSample((0.6072,),(0.1601,))
    '''

    trainer=BackpropTrainer(net,ds,verbose=True)

    trainer.trainUntilConvergence(dataset=ds,verbose=True)

    #print trainer.testOnData(dataset=ds,verbose=True)

    d=net.activate((inp,))
    print d[0]
    return d[0]

'''
def main():
    print "Enter max amplitude: "
    inp=input()
    out=trainNetwork(inp)
    thr=out[0]
    print thr
    #return thr


if __name__ == '__main__':main()
'''