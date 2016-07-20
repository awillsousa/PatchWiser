# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016

@author: antoniosousa

Realiza a extraçao de atributos e armazena num arquivo em formato SVM Light
"""
import sys
import os
import extrator as ex
import arquivos as arq
import numpy as np
from sklearn.datasets import dump_svmlight_file

diretorio = "/home/willian/basesML/bases_cancer/treino/"
extratores = ['lbp', 'glcm', 'pftas', 'todos']
padrao='*.png'

def extrai_base_atributos(diretorio, extrator='pftas', padrao='*.png', desc_arquivo='', descarte=False):    
   
   atributos, rotulos = ex.extrai_atributos(diretorio, extrator, padrao, descarte)  
   arq_saida = diretorio + "base_"+extrator+"_"+desc_arquivo+".svm"     
   dump_svmlight_file(atributos, rotulos, arq_saida)          

'''
Chamada programa principal
'''

if __name__ == "__main__":
  if len(sys.argv) < 1:
        sys.exit("Uso: extracTHis.py <diretorio> <metodo> [-d (usar descarte)]")
  
  arg1 = sys.argv[1]
  if not (os.path.isdir(arg1)):
      sys.exit("Erro: Caminho incorreto ou o diretorio nao existe.")
      
  if len(sys.argv) < 2:
      arg2 = 'lbp'
  else:
      arg2 = sys.argv[2]
      try:
          i = extratores.index(arg2)
      except ValueError:
          sys.exit("Erro: Utilize um dos seguintes extratores " + str(extratores))
  
  arg3 = False          
  if len(sys.argv) > 3 and (sys.argv[3] == "-d"):
      arg3 = True
  else:
      sys.exit("Parametro de descarte de patches incorreto! Utilize -d.")      
        
  extrai_base_atributos(arg1, arg2, arg3)

