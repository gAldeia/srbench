from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union

import os
import subprocess
import pandas as pd
import numpy as np
import time

this_dir = os.path.dirname(os.path.realpath(__file__))

class GSGPRegressor(BaseEstimator):




  def __init__(self,  g=100, popsize=1000, rt_mut=0.5, rt_cross=0.5, 
               max_len=10, n_jobs=1):
    env = dict(os.environ)
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len
    self.trainsize=-1
    self.nvar=-1
    self.exe_name = 'GP'

  def line_prepender(self,filename, line):
    with open(filename, 'r+') as f:
      content = f.read()
      f.seek(0, 0)
      f.write(line.rstrip('\r\n') + '\n' + content)


  def fit(self, X_train, y_train, sample_weight=None):
    self.X_train=X_train
    self.y_train=y_train
    self.y_test=y_train # this is a hack just to run the executable
    text='''population_size={}
max_number_generations={}
init_type = 2
p_crossover={}
p_mutation={}
max_depth_creation={}
tournament_size= 4
zero_depth = 0
mutation_step = 1
num_random_constants = 0
min_random_constant = -100
max_random_constant = 100
minimization_problem = 1
random_tree = 500
expression_file = 0
USE_TEST_SET = 0
'''.format(self.popsize, self.g, self.rt_cross,self.rt_mut,self.max_len)

    self.dataset = this_dir + '/tmp_data_' + str(np.random.randint(2**15-1))
    self.dataset_short = self.dataset.split('/')[-1]
    ffile=open(self.dataset+"-configuration.ini","w")
    ffile.write(text)


  def predict(self, X_test):

    # train data
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train
    data.to_csv(self.dataset+"_train",
                header=None, index=None, sep='\t')
    trainsize=self.X_train.shape[0]
    nvar=self.X_train.shape[1]
    self.line_prepender(self.dataset+'_train',str(trainsize)+'\n')
    self.line_prepender(self.dataset+'_train',str(nvar)+'\n')
    time.sleep(1)

    # test data dummy: required by GSGP to be passed during training.
    # I refuse to pass the target into this exe on the test data during train, 
    # so I'm making a dummy random target.
    datat_dummy=pd.DataFrame(X_test)
    datat_dummy['target']=np.random.rand(len(X_test))
    datat_dummy.to_csv(self.dataset+"_test_dummy",
                  header=None, index=None, sep='\t')
    testsize=X_test.shape[0]
    nvar=X_test.shape[1]
    time.sleep(1)
    self.line_prepender(self.dataset+'_test_dummy',str(X_test.shape[0])+'\n')
    self.line_prepender(self.dataset+'_test_dummy',str(X_test.shape[1])+'\n')
    time.sleep(1)

    #test data 2: the test data, without target.
    datat2 = datat_dummy.drop('target',axis=1)
    datat2.to_csv(self.dataset+"_test",
                  header=None, index=None, sep='\t')

    #do training
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 0/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 0/g' "
                     +self.dataset+"-configuration.ini"],shell=True)

    print('cmd:',' '.join([this_dir + '/' + self.exe_name, 
                     ' -train_file '+ self.dataset+'_train',
                     '-test_file ', self.dataset+'_test_dummy',
                     " -name "+ self.dataset]))

    subprocess.call(' '.join([this_dir + '/' + self.exe_name, 
                     ' -train_file '+ self.dataset+'_train',
                     '-test_file ', self.dataset+'_test_dummy',
                     " -name "+ self.dataset]),
                    shell=True)
    time.sleep(1)
    #do testing
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 1/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 1/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(' '.join([this_dir + '/' + self.exe_name, 
                     '-test_file',self.dataset+ '_test',
                     '-name ', self.dataset]),
                    shell=True)
    time.sleep(1) # without this, the output file sometimes DNE
    y_pred=[]
    with open(self.dataset+'-evaluation_on_unseen_data.txt','r') as f:
      for line in f:
        y_pred.append(float(line.strip()))
    # y_pred=y_pred[:-1]
    print('y_pred:',len(y_pred),y_pred)
    print('X len',len(X_test))
    assert(len(y_pred) == len(X_test))

    os.remove(self.dataset+"_train")
    os.remove(self.dataset+"_test")
    os.remove(self.dataset+"_test_dummy")
    return y_pred
