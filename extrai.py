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

def main(diretorio, extrator='lbp', descarte=False, padrao='*.png'):    
    lista_imagens = arq.busca_arquivos(diretorio, padrao)
    
    if (extrator == 'lbp' or extrator == 'todos'):
        atributos, rotulos = ex.extrai_lbp(lista_imagens, descarte)
        arq_saida = diretorio + "D-base_LBP.svm"  
        dump_svmlight_file(atributos, rotulos, arq_saida)          
    if (extrator == 'pftas' or extrator == 'todos'):
        atributos, rotulos = ex.extrai_pftas(lista_imagens, descarte)  
        arq_saida = diretorio + "D-base_PFTAS.svm" 
        dump_svmlight_file(atributos, rotulos, arq_saida)     
    if (extrator == 'glcm' or extrator == 'todos'):
        atributos, rotulos = ex.extrai_haralick(lista_imagens, descarte)  
        arq_saida = diretorio + "D-base_GLCM.svm" 
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
        
  main(arg1, arg2, arg3)

