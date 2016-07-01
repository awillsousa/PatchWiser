# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:31:25 2016
@author: antoniosousa

Programa para realizar treinamento e classificação de imagens de cancer
(tumores malignos e benignos)
"""

import sys
import time
from os import path
import extrator as ex
import arquivos as arq
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
import pylab as pl
from optparse import OptionParser

#BASETR_LBP = "/home/willian/basesML/bases_cancer/base_LBP.svm"
#BASETR_PFTAS = "/home/willian/basesML/bases_cancer/base_PFTAS.svm"
#BASETR_GLCM = "/home/willian/basesML/bases_cancer/base_GLCM.svm"
BASETR_LBP = "base_LBP.svm"
BASETR_PFTAS = "base_PFTAS.svm"
BASETR_GLCM = "base_GLCM.svm"
BASES = {'lbp':BASETR_LBP, 'pftas':BASETR_PFTAS, 'glcm':BASETR_GLCM}
EXTRATORES = ['lbp', 'pftas', 'glcm']
CLFS = {'knn':('KNN', KNeighborsClassifier(1)), 
        'svm':('SVM', SVC(gamma=0.5, C=32)),
        'dt':('Árvore de Decisão', DecisionTreeClassifier(max_depth=5)),
        'qda':('QDA', QuadraticDiscriminantAnalysis())}

FUSAO = ['voto', 'soma', 'produto', 'serie']     

# parametros-padrão do classificador SVM
SVM_C = 32
SVM_G = 0.5

'''
Realiza o treinamento de uma base
'''
def treina(id_clf='knn', metodo='lbp'):
    
    clf = get_clf(id_clf)    
    print('Usando: ' + get_desc_clf(id_clf) + ' com ' + metodo)
            
    
    # Carrega a base de treinamento
    base_tr = BASES[metodo]    
    atrib_dados, rotulos_dados = load_svmlight_file(base_tr)
    atrib_tr, atrib_ts, rotulos_tr, rotulos_ts =  cross_validation.train_test_split(atrib_dados, rotulos_dados, test_size=0.3)    
    
    if id_clf == "svm":
        clf = SVC(SVM_G, SVM_G)
    
    # Para o classificador QDA deve-se usar uma matriz densa
    if id_clf == "qda":    
        atrib_tr = atrib_tr.toarray()    
    
    # Treina o classificador passado
    clf.fit(atrib_tr, rotulos_tr)
 

    # Para o classificador QDA deve-se usar uma matriz densa
    if id_clf == "qda":    
        atrib_ts = atrib_ts.toarray() 
        
    # predicao do classificador            
    rotulos_pred = clf.predict(atrib_ts)

    # mostra o resultado do classificador na base de teste
    print (clf.score(atrib_ts, rotulos_ts))
            
    # cria a matriz de confusao
    cm = confusion_matrix(rotulos_ts, rotulos_pred)
    print (cm) 
    pl.matshow(cm) 
    pl.colorbar() 
    pl.show() 

'''
Classifica uma base de imagens
'''
def classifica(base, base_tr, id_clf='knn', metodo='lbp', svm_c=SVM_C, svm_g=SVM_G):
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
    
    # Carrega a base de treinamento        
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    # Para o classificador QDA deve-se usar uma matriz densa
    if id_clf == "qda":    
        atrib_tr = atrib_tr.toarray()  
    
    # Treina o classificador passado
    clf = get_clf(id_clf)    
    clf.fit(atrib_tr, rotulos_tr)
        
    rotulos_ts = []
    rotulos_pred = []
    # classifica as imagens uma a uma    
    for imagem in lista_imagens:
        # recupera o rotulo real da imagem (total)
        classe, _ = ex.classe_arquivo(imagem)                
        rotulos_ts.append(ex.CLASSES[classe])
        
        # extrai os patches da imagem 
        if (metodo == 'lbp'):            
            atrib_vl, rotulos_vl = ex.extrai_lbp([imagem])
        elif (metodo == 'pftas'):
            atrib_vl, rotulos_vl = ex.extrai_pftas([imagem])
        elif (metodo == 'glcm'):
            atrib_vl, rotulos_vl = ex.extrai_haralick([imagem])
        
        if len(atrib_vl) > 0:
            # Para o classificador QDA deve-se usar uma matriz densa
            if id_clf == "qda":    
                atrib_vl = np.asarray(atrib_vl)
                
            # predicao do classificador para o conjunto de patches        
            ls_preds = clf.predict(atrib_vl) 
            #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
            ls_preds = np.asarray(ls_preds, dtype='int32')                        
            conta = np.bincount(ls_preds)
            #print('conta ' + str(conta))
            rotulos_pred.append(np.argmax(conta))
    
    return (rotulos_ts, rotulos_pred)

'''
Classifica uma base de imagens
'''
def classifica_serie(lista_imagens, base_tr, id_clf='knn', metodo='lbp', svm_c=SVM_C, svm_g=SVM_G):
    
    # Carrega a base de treinamento        
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    # Para o classificador QDA deve-se usar uma matriz densa
    if id_clf == "qda":    
        atrib_tr = atrib_tr.toarray()  
    
    # Treina o classificador passado
    clf = get_clf(id_clf)    
    clf.fit(atrib_tr, rotulos_tr)
            
    lista_clfd = []
    # classifica as imagens uma a uma    
    for imagem in lista_imagens:
        # recupera o rotulo real da imagem (total)
        classe, _ = ex.classe_arquivo(imagem)  
        r_ts = ex.CLASSES[classe] 
        
        # extrai os patches da imagem 
        if (metodo == 'lbp'):            
            atrib_vl, rotulos_vl = ex.extrai_lbp([imagem])
        elif (metodo == 'pftas'):
            atrib_vl, rotulos_vl = ex.extrai_pftas([imagem])
        elif (metodo == 'glcm'):
            atrib_vl, rotulos_vl = ex.extrai_haralick([imagem])
        
        # Para o classificador QDA deve-se usar uma matriz densa
        if id_clf == "qda":    
            atrib_vl = np.asarray(atrib_vl)
            
        # predicao do classificador para o conjunto de patches        
        ls_preds = clf.predict(atrib_vl) 
        #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
        ls_preds = np.asarray(ls_preds, dtype='int32')                        
        conta = np.bincount(ls_preds)
        r_pred = np.argmax(conta)
        
        lista_clfd.append((imagem,r_pred, r_ts))
            
    return (lista_clfd)

'''
Prediz os labels de uma base de imagens
'''
def prediz(lista_imagens, base_tr, id_clf='knn', metodo='lbp', svm_c=SVM_C, svm_g=SVM_G):
    
    # Carrega a base de treinamento        
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    # Para o classificador QDA deve-se usar uma matriz densa 
    if id_clf == "qda":    
        atrib_tr = atrib_tr.toarray()  
    
    # Treina o classificador passado
    clf = get_clf(id_clf)    
    clf.fit(atrib_tr, rotulos_tr)
    
    rotulos_pred = []
    # classifica as imagens uma a uma    
    for imagem in lista_imagens:                
        # extrai os patches da imagem 
        if (metodo == 'lbp'):            
            atrib_vl, rotulos_vl = ex.extrai_lbp([imagem])
        elif (metodo == 'pftas'):
            atrib_vl, rotulos_vl = ex.extrai_pftas([imagem])
        elif (metodo == 'glcm'):
            atrib_vl, rotulos_vl = ex.extrai_haralick([imagem])
        
        # Para o classificador QDA deve-se usar uma matriz densa
        if id_clf == "qda":    
            atrib_vl = np.asarray(atrib_vl)
            
        # predicao do classificador para o conjunto de patches        
        ls_preds = clf.predict(atrib_vl) 
        #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
        ls_preds = np.asarray(ls_preds, dtype='int32')                        
        conta = np.bincount(ls_preds)        
        rotulos_pred.append(np.argmax(conta))
    
    return (rotulos_pred)

'''
Retorna os labels de uma base de imagens
'''
def get_labels(lista_imagens):
    
    rotulos_ts = []
    # recupera os labels das imagens uma a uma    
    for imagem in lista_imagens:
        # recupera o rotulo real da imagem (total)
        classe, _ = ex.classe_arquivo(imagem)                
        rotulos_ts.append(ex.CLASSES[classe])
        
    return (rotulos_ts)

'''
Realiza a fusao de 2 ou mais classificadores por voto (hard)
'''
def fusao_voto(base, base_tr, clfs, metodo='pftas'): 
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
       
    # Carrega rotulos de teste        
    rotulos_ts = get_labels(lista_imagens) 
    
    preds_clfs = []
    for id_clf in clfs:    
        # para cada classificador, armazena as classificacoes
        r_pred = prediz(lista_imagens, base_tr, id_clf, metodo, svm_c=SVM_C, svm_g=SVM_G)
        preds_clfs.append(r_pred)
    
    preds_clfs = np.asarray(preds_clfs)    
    rotulos_pred = []
    # Faz a fusao por voto
    for p in preds_clfs.T:
        ls_preds = np.asarray(p, dtype='int32')                        
        conta = np.bincount(ls_preds)        
        rotulos_pred.append(np.argmax(conta))
    
    return (rotulos_ts, rotulos_pred)#, preds_clfs) 

'''
Realiza a fusao de 2 classificadores por voto e avalia
onde há divergencia. 
'''
def fusao_voto_sec(base, base_tr, clfs, metodo='pftas'): 
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
       
    # Carrega rotulos de teste        
    rotulos_ts = get_labels(lista_imagens)
    
    preds_clfs = []
    for id_clf in clfs:    
        # para cada classificador, armazena as classificacoes
        r_pred = prediz(lista_imagens, base_tr, id_clf, metodo, svm_c=SVM_C, svm_g=SVM_G)
        preds_clfs.append(r_pred)
    
    preds_clfs = np.asarray(preds_clfs)    
    rotulos_pred = []
    # Faz a fusao por voto
    for p in preds_clfs.T:
        ls_preds = np.asarray(p, dtype='int32')                        
        conta = np.bincount(ls_preds)        
        rotulos_pred.append(np.argmax(conta))
    
    return (rotulos_ts, rotulos_pred,preds_clfs) 

'''
Realiza a fusao de 2 ou mais classificadores por soma
'''
def fusao_soma(base, base_tr, clfs, metodo='pftas'): 
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
       
    # Carrega a base de treinamento        
    rotulos_ts = get_labels(lista_imagens)
    rotulos_pred = rotulos_ts
      
    return (rotulos_ts, rotulos_pred) 
    
''' 
Realiza a fusao de 2 ou mais classificadores por voto (hard)
''' 
def fusao_produto(base, base_tr, clfs, metodo='pftas'): 
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
       
    # Carrega a base de treinamento        
    rotulos_ts = get_labels(lista_imagens)
    rotulos_pred = rotulos_ts
      
    return (rotulos_ts, rotulos_pred) 

'''
Realiza a fusao de 2 ou mais classificadores em serie
'''
def fusao_serie(base, base_tr, clfs, metodo='pftas'): 
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
    imgs_clfds = []   
    
    # Lista de rotulos preditos e rotulos reais
    rotulos_pred = []  
    rotulos_ts = []
    
    for id_clf in clfs:
        print("Lista a classificar: " + str(len(lista_imagens)))
        # para cada classificador, armazena as classificacoes
        imgs_clfds = classifica_serie(lista_imagens, base_tr, id_clf, metodo)
        lista_imagens = []        
        for c in imgs_clfds:
            if (c[1] != c[2]):  # apenas as imagens que nao foram classificadas corretamente
                lista_imagens.append(c[0]) 
            else:
                rotulos_pred.append(c[1])
                rotulos_ts.append(c[2])
    
    # acrescenta na lista de rotulos predictos e reais aqueles que nao foram 
    # classificados corretamente
    for c in imgs_clfds:
        if (c[1] != c[2]):                 
                rotulos_pred.append(c[1])
                rotulos_ts.append(c[2])
      
    return (rotulos_ts, rotulos_pred) 

def get_clf(nome_clf):
    c = CLFS[nome_clf]    
    return (c[1])

def get_desc_clf(nome_clf):
    c = CLFS[nome_clf]     
    return (c[0])

def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False  

'''
Programa Principal
'''
#def main(): 
inicio = time.time()

parser = OptionParser()
parser.add_option("-T", "--base-treino", dest="base_treino",
                  help="Localização do arquivo da base de treino")
parser.add_option("-t", "--base-teste", dest="base_teste",
                  help="Localização do arquivo da base de teste") 
#parser.add_option("-d", "--dir-clf", dest="dir_clf",
#                  help="Diretório de imagens a serem classificadas")
parser.add_option("-C", "--clf", dest="opt_clf",
                  default='dt',
                  help="Lista de classificadores a serem utilizados. [dt, qda, svm, knn]")
parser.add_option("-f", "--fusao", dest="fusao_clf",
                  help="Método de fusão de classificadores a utilizar. [voto, soma, produto]")
parser.add_option("-c", "--svm-c", dest="svm_c",                      
                  help="Valor do parâmetro C do SVM")
parser.add_option("-g", "--svm-g", dest="svm_g",                      
                  help="Valor do parâmetro gamma do SVM")                      
 
(options, args) = parser.parse_args()

# verifica se a base de treino e de teste passadas existem
if not (path.isfile(options.base_treino)):
  sys.exit("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")    
if not (path.isdir(options.base_teste)):
  sys.exit("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")  
    
opt_clfs = options.opt_clf.split(',')     
# passado mais de um classificador sem um metodo de fusao definido    
if (len(opt_clfs) > 1):
    if not(existe_opt(parser,'fusao_clf')):
        sys.exit("Erro: Metodo de fusao de classificadores ausente.")
    # verifica se os metodo de fusao definido é válido
    if not(options.fusao_clf in FUSAO):
       sys.exit("Erro: Metodo de fusao desconhecido. Valores aceitos: " + str(FUSAO)) 
    
# verifica se os classificadores passados são validos
for c in opt_clfs:
    if not(CLFS[c]):
       sys.exit("Erro: Classificador desconhecido. Valores aceitos: " + str(CLFS.keys)) 


resultados = []
# metodo a ser utilizado
s = options.base_treino.split('_')
s = s[1].split('.')
metodo = s[0].lower()

if (len(opt_clfs) > 1): # ha mais de uma classificador para fundir
   if options.fusao_clf == 'voto':
       r_tst,r_pred = fusao_voto(options.base_teste, options.base_treino, opt_clfs, metodo)
   elif options.fusao_clf == 'soma':
        sys.exit("Ainda não implementado!")       
       #r_tst,r_pred = fusao_soma(options.base_teste, options.base_treino, opt_clfs, metodo)
   elif options.fusao_clf == 'produto':
        sys.exit("Ainda não implementado!")       
       #r_tst,r_pred = fusao_produto(options.base_teste, options.base_treino, opt_clfs, metodo)
   else: #padrao é serie
       r_tst,r_pred = fusao_serie(options.base_teste, options.base_treino, opt_clfs, metodo)      
else: # apenas um classificador        
    if opt_clfs[0] == 'svm':
        if (options.svm_c):
            SVM_C = float(options.svm_c)
        if (options.svm_g):
            SVM_G = float(options.svm_g)
            
    r_tst,r_pred = classifica(options.base_teste, options.base_treino, opt_clfs[0], metodo, SVM_C, SVM_G)
    #r_tst,r_pred = classifica(options.base_teste, options.base_treino, opt_clfs[0])

# cria as matrizes de confusao
cm = confusion_matrix(r_tst, r_pred)
print("Matriz de confusao: ")
print (cm) 
#pl.matshow(cm)
#pl.colorbar()
#pl.show()
 
# exibe a taxa de classificacao
r_pred = np.asarray(r_pred)
r_tst = np.asarray(r_tst)
taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
print("Taxa de Classificação: %f " % (taxa_clf)) 
 
# Exibe o tempo de execucao
print ("Tempo total de execução: " +str(round(time.time() - inicio, 3)))
 
'''
Chamada programa principal
''' 
   
#if __name__ == "__main__":	
#	main()
