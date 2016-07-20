# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Testes de classificação
"""

import datetime
from os import path
from time import time
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from optparse import OptionParser
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cv2
import sys
import math
import numpy as np
import mahotas as mh
import extrator as ex
import arquivos as arq
import binarypattern as bp
import matplotlib.pyplot as plt
import sliding_window as sw


# Constantes
LIMIAR_DIVERGENCIA=0.25
N_DIV = 2
SVM_C = 32
SVM_G = 0.5
CLFS = {'knn':('KNN', KNeighborsClassifier(1)), 
        'svm':('SVM', SVC(gamma=0.5, C=32)),
        'dt':('Árvore de Decisão', DecisionTreeClassifier(max_depth=5)),
        'qda':('QDA', QuadraticDiscriminantAnalysis())}
FUSAO = ['voto', 'soma', 'produto', 'serie']     
CLASSES = {'B':0, 'M':1}


## LOG LOG LOG ##
hoje = datetime.datetime.today()
formato = "%a-%b-%d-%H_%M_%S"
arq_log = hoje.strftime(formato)+".log"
logfile = open("./logs/"+"npatches-"+arq_log, "w")


'''
Escreve no console e no arquivo de log da aplicacao
'''
def log(texto):    
    # Exibe no console e escreve também no arquivo de log
    print(texto)
    print(texto, file=logfile)
    

def get_clf(nome_clf): 
    c = CLFS[nome_clf]    
    return (c[1]) 

def get_desc_clf(nome_clf):
    c = CLFS[nome_clf]     
    return (c[0])

def cria_patches3(imagem, lado, rgb=False):          
    #imagem = np.asarray(imagem)     
    window_size = (lado,lado) if not(rgb) else (lado,lado,3)
    print("Lado: "+ str(lado) + " Window: "+ str(window_size) + " Imagem: " + str(imagem.shape))
    windows = sw.sliding_window_nd(imagem, window_size)  
    print ("Cria_patches3:" + str(windows.shape))
    return (windows)


'''
Utilizado para testar a funcao de criacao de patches
'''
def exibe_cria_patches(diretorio, n_divs=3):
    lista_imagens = arq.busca_arquivos(diretorio, "*.png")
    #converte para escala de cinza
    for arquivo in lista_imagens:
        img = mh.imread(arquivo)            
        
        #patches = bp.patches(img, TAM_PATCH, rgb=True)
        img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
        #img_cinza = bp.limpa_imagem(img_cinza)                    
        #patches_cinza = bp.cria_patches(img_cinza, TAM_PATCH)        
        #patches_cinza = bp.patches(img_cinza, TAM_PATCH, rgb=False)        
        
        
        # exibe a imagem original    
        plt.imshow(img)                
        #patches = cria_patches3(img, n_divs, rgb=True)
        patches = cria_patches3(img, 32, rgb=True)
        print("Total de patches: %f", len(patches))
        print("Tamanho do patch: %i", patches[0].shape)
                    
        #y = int(math.sqrt(4**n_divs))
        y = int(math.sqrt(len(patches)))
        x = y 
        
        fig,axes = plt.subplots(x,y) 
        for i, ax in enumerate(axes.ravel()): 
            if (i == len(patches)):
                break;
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            im = ax.imshow(patches[i],'gray')         
        plt.show()    

   
'''
Para cada imagem da lista de imagens passada, divide a imagem em patches
e para cada patch aplica um extrator de descritor de textura 
'''
def extrai(arquivo, n_div):
    atributos = []    
    rotulos = []    
        
    # recupera do nome do arquivo a sua classe 
    classe, _ = ex.classe_arquivo(arquivo)
            
    #converte para escala de cinza
    img = mh.imread(arquivo)            
    patches = cria_patches3(img, n_div, rgb=True)        
    
    # calcula o histograma de cada um dos patches    
    for i, patch in enumerate(patches):        
        p_pftas = mh.features.pftas(patch)           
        atributos.append(p_pftas)  
        rotulos.append(ex.CLASSES[classe])                             
    
    return (atributos,rotulos)    

def gera_arqs_treino(diretorio, n_divs):
    
    arqs_treino = {}    # dicionario de arquivos de treino por qtd de patches
    for n in range(0,n_divs):        
        arq_saida = diretorio + "base_PFTAS_"+str(n)+"_divs.svm" 
        arqs_treino[n] = arq_saida
    
    return (arqs_treino)

'''
Recebe uma lista de imagens e extrai patches dessas imagens de acordo com a 
quantidade de divisões passadas. Por exemplo, 1 divisão, dividirá a imagem em 
4 partes. 2 divisões, dividirá a imagem em 16 partes e assim por diante. Caso
seja utilizada 0 divisões, será considerada a imagem inteira. 
Para cada uma das divisões consideradas, é gerada uma base de atributos extraidos
utilizando PFTAS.  
'''
def extrai_patches_imgs(lista_imagens, diretorio, n_divs=5):
    # Cria os dicionarios dos atributos e rotulos por qtd de patches
    atributos = {}      # dicionario de atributos por qtd de patches
    rotulos = {}        # dicionario de rotulos por qtd de patches    
    for n in range(0,n_divs):        
        atributos[n] = []    
        rotulos[n] = []     
        
    arqs_treino = gera_arqs_treino(diretorio, n_divs)    
    ##  INICIO DO PROCESSO DE EXTRACAO DE ATRIBUTOS    
    
    for arq_imagem in lista_imagens:
        imagem = mh.imread(arq_imagem)
        classe, _ = ex.classe_arquivo(arq_imagem)
        
        # Extrai os atributos e gera os arquivos dos patches da base de treino
        atrs,rots,patches = extrai_pftas_patches(imagem, classe, 0)    
        atributos_img = atributos.get(0)
        atributos_img += atrs     
        rotulos_img = rotulos.get(0)
        rotulos_img += rots
        
        # Executa para todos os tamanhos de patches e acumula as listas de atributos
        # e os rotulos de cada um dos patches gerados
        # A vantagem desse metodo é realizar apenas uma leitura em disco e evitar 
        # excesso de overhead ao dividir a imagem em imagens menores (patches)
        # pois ja divide a imagem e extrai atributos para todos os tamanhos de patches
        for n in range(1, n_divs): 
            novos_patches = []
                               
            for img in patches:         
                atrs, rots, p = extrai_pftas_patches(img, classe)  
                atributos_img = atributos.get(n)
                atributos_img += atrs     
                rotulos_img = rotulos.get(n)
                rotulos_img += rots
                
                novos_patches += p
            
            patches = novos_patches
        
    for n in range(0, n_divs):             
        dump_svmlight_file(atributos.get(n), rotulos.get(n), arqs_treino.get(n))
        atributos[n] = []    
        rotulos[n] = []

'''
Exibe o tempo passado a partir do inicio passado e retorna o tempo atual
em uma nova variável
'''    
def exibe_tempo(inicio, desc=""):
    fim = round(time() - inicio, 3)
    print ("Tempo total de execução ["+desc+"]: " +str(fim))
    
    return(time())     
 
  
'''
Extrai atributos dos patches de imagem passada.
Inicialmente, divide a imagem em patches para em seguida
aplicar ao método PFTAS
'''  
def extrai_pftas_patches_n(img, classe, lado=2):
    atributos = []    
    rotulos = []
        
    patches = cria_patches3(img, lado, rgb=True)
    
    # calcula o histograma de cada um dos patches    
    for i, patch in enumerate(patches):
        p_pftas = mh.features.pftas(patch)           
        atributos.append(p_pftas)  
        rotulos.append(ex.CLASSES[classe])                             
        
    return (atributos,rotulos)
  
'''
Executa a extracao de atributos de imagens, utilizando patches
de diversos tamanhos. Inicial 2x2, gerando patches incrementais
3x3,4x4, até 128x128
'''    
def executa_extracao_n(base_treino, metodo, n=1):
    inicio = time()    
    
    lista_imagens = arq.busca_arquivos(base_treino, "*.png")
    n_imgs_treino = len(lista_imagens)
    
    for lado in range(8,n+1,4):
        atributos = []    
        rotulos = []     
            
        arq_treino = base_treino + "base_PFTAS_"+str(lado)+"x"+str(lado)+".svm"
        ##  INICIO DO PROCESSO DE EXTRACAO DE ATRIBUTOS    
        
        for arq_imagem in lista_imagens: 
            print("Arquivo: " + arq_imagem)
            imagem = mh.imread(arq_imagem) 
            if (imagem != None):
                classe, _ = ex.classe_arquivo(arq_imagem)             
                print("executa_extracao_n - shape imagem:" + str(imagem.shape))
                # Extrai os atributos e gera os arquivos dos patches da base de treino
                atrs,rots = extrai_pftas_patches_n(imagem, classe, lado)                            
                atributos += atrs
                rotulos += rots
        
        dump_svmlight_file(atributos, rotulos, arq_treino)
    
    log("Extraidos atributos da base " + base_treino + " utilizando " + metodo + "\n para " + str(n_imgs_treino) + "imagens") 
  
    # Exibe o tempo de execução    
    log(str(time()-inicio) + "EXTRAÇÃO")     
   
'''
Executa a classificacao de uma base de imagens 
'''
def classifica_n(base, atrib_tr, rotulos_tr, id_clf, n, svm_c=SVM_C, svm_g=SVM_G):
    
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
    
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
        log("Extraindo patches da imagem " + imagem)        
        atrib_vl, rotulos_vl = extrai(imagem, n)
        
        
        if len(atrib_vl) > 0:            
            log("Predizendo classe da imagem")
            # predicao do classificador para o conjunto de patches        
            ls_preds = clf.predict(atrib_vl) 
            ls_preds = np.asarray(ls_preds, dtype='int32')                        
            conta = np.bincount(ls_preds)
            log('Contagem classes patches: ' + str(conta))
            cl = np.argmax(conta)
            rotulos_pred.append(cl)
            log ('Classe: ' + str(cl))
    return (rotulos_ts, rotulos_pred)
   
   
'''
Executa classificacao da base de testes utilizando uma base de treino
de imagens que foram diviidas em 4**n patches
'''    
def executa_classificacao_n(base_teste, base_treino, n=1):
    inicio1 = time()    
    
    tamanhos = []
    taxas = []
    matrizes = []
    tempos = []
    c = "svm" 
    
    try:
        for lado in range(8,n+1,4):
            # Carrega a base de treinamento         
            base_tr = base_treino + "base_PFTAS_"+str(lado)+"x"+str(lado)+".svm"
            atrib_tr = None
            rotulos_tr = None    
            atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
            
            # Classifica a base de testes            
            inicio = time()           
            log("Classificando para " + c + " usando patches de "+str(lado)+"x"+str(lado))
                    
            r_tst,r_pred = classifica_n(base_teste, atrib_tr, rotulos_tr, c, lado)
            
            # cria as matrizes de confusao
            cm = confusion_matrix(r_tst, r_pred)            
            
            # exibe a taxa de classificacao
            r_pred = np.asarray(r_pred)
            r_tst = np.asarray(r_tst)
            taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100            
            
            # armazena os resultados 
            tamanhos.append(lado) 
            taxas.append(taxa_clf) 
            matrizes.append(cm) 
            tempos.append(time()-inicio) 
            inicio = exibe_tempo(inicio, "CLASSIFICACAO") 
        
    except FileNotFoundError as fnf:
        log("I/O error({0}): {1}".format(fnf.errno, fnf.strerror))
    except TypeError as te:
        log("Type Error({0}): {1}".format(te.errno, te.strerror))
    except:
        log("ERRO ou PROBLEMA desconhecido no processo de classificação.") 
        print ("Unexpected error:", sys.exc_info()[0])
        raise           
        
    # Exibe o tempo de execução    
    log("Tempo: " + str(time()-inicio1) + " CLASSIFICACAO para " + str(4**n) + " patches")     
    
    ## FIM DO PROCESSO DE CLASSIFICACAO
    
    # exibe os dados obtidos
    for t,tx,mc in zip(tamanhos,taxas,matrizes):
        log("\nTamanho: " + str(t) + " Taxa Reconhecimento: " + str(tx))        
        log("Matriz de Confusão: \n" + str(mc))        
        
     # plota grafico de resultados [reconhecimento vs tam. patch]
    arq_grafico = base_treino + "PFTAS_"+str(lado)+"x"+str(lado)+".pdf"
    plota_grafico(tamanhos, taxas, arq_grafico, tituloX="Tam. Patch", tituloY="Tx. Reconhecimento")
    
    arq_grafico_tempo = base_treino + "Tempos_" +str(lado)+"x"+str(lado)+".pdf"
    plota_grafico(tamanhos, tempos, arq_grafico_tempo, tituloX="Num. Patches", tituloY="Tempo")


###############################################################################

def plota_grafico(dadosX, dadosY, arquivo="grafico.pdf", titulo="", tituloX="X", tituloY="Y", ):
    # plota grafico de resultados [reconhecimento vs tam. patch]
    plt.plot(dadosX, dadosY)
    
    # anota os pontos de classificacao
    for x,y in zip(dadosX,dadosY):        
        plt.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
        plt.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
        plt.scatter([x,],[y,], 50, color ='red')
        
    plt.ylabel(tituloY)
    plt.xlabel(tituloX)
    if (titulo == ""):
        titulo = tituloX + " vs " + tituloY
    plt.title(titulo)
    
    # set axis limits
    plt.xlim(0.0, max(dadosX))
    max_y = max(dadosY)
    if max_y < 100:
        max_y = 100
        
    plt.ylim(0.0, max_y)
    #plt.savefig(arquivo, bbox_inches='tight')
    plt.savefig(arquivo)
    plt.clf()
       

###############################################################################

def main():
    treino="/home/willian/basesML/bases_cancer/folds-spanhol/mkfold/fold5/train/40X/"
    teste="/home/willian/basesML/bases_cancer/folds-spanhol/mkfold/fold5/test/40X/"
    #treino="/home/willian/basesML/bases_cancer/min_treino2/"    
    #teste="/home/willian/basesML/bases_cancer/min_teste/"    
    n=128
    metodo="pftas"
    
    executa_extracao_n(treino, metodo, n)
    #executa_classificacao_n(teste,treino,n)
   
# Programa principal 
if __name__ == "__main__":	
	main()