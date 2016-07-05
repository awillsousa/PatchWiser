# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016

@author: antoniosousa

Testes de classificação
"""

from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from time import time
import binarypattern as bp
import arquivos as arq
import numpy as np
import extrator as ex
import mahotas as mh
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Constantes
LIMIAR_DIVERGENCIA=0.25
N_DIV = 2
SVM_C = 32
SVM_G = 0.5
CLFS = {'knn':('KNN', KNeighborsClassifier(1)), 
        'svm':('SVM', SVC(gamma=0.5, C=32)),
        'dt':('Árvore de Decisão', DecisionTreeClassifier(max_depth=5)),
        'qda':('QDA', QuadraticDiscriminantAnalysis())}

def get_clf(nome_clf):
    c = CLFS[nome_clf]    
    return (c[1])

def get_desc_clf(nome_clf):
    c = CLFS[nome_clf]     
    return (c[0])

def cria_patches(imagens, n_div, rgb=False): 
    # condicao de parada da recursao    
    if (n_div == 0):
        return (imagens);
        
    colecao = []
    for imagem in imagens:    
        if not(rgb):
            l, h = imagem.shape  # retorna largura e altura da imagem
        
            # calcula os valores medios
            h_m = int(math.ceil(h/2))             
            l_m = int(math.ceil(l/2))
            print ("l_m, h_m: ", str((l_m, h_m)))
            # divide a imagem passada em 4 patches        
            # patch 0,0            
            m = np.copy(imagem[:l_m,:h_m])                
            colecao.append(m)                    
            # patch 1,0            
            m = np.copy(imagem[:l_m,h_m:])                
            colecao.append(m)                    
            # patch 0,1        
            m = np.copy(imagem[l_m:,:h_m])                
            colecao.append(m)                    
            # patch 1,1
            m = np.copy(imagem[l_m:,h_m:])                
            colecao.append(m)                    
        else:
            l, h, _ = imagem.shape  # retorna largura e altura da imagem
            # calcula os valores medios
            h_m = int(math.ceil(h/2))             
            l_m = int(math.ceil(l/2))   
            #print ("l_m, h_m: ", str((l_m, h_m)))
            # divide a imagem passada em 4 patches        
            # patch 0,0            
            m = np.copy(imagem[:l_m:,:h_m:])                
            colecao.append(m)                    
            # patch 1,0            
            m = np.copy(imagem[:l_m:,h_m::])                
            colecao.append(m)                    
            # patch 0,1        
            m = np.copy(imagem[l_m::,:h_m:])                
            colecao.append(m)                    
            # patch 1,1
            m = np.copy(imagem[l_m::,h_m::])                
            colecao.append(m)
            
    n_div -= 1
    return (cria_patches(colecao, n_div, rgb))    

def cria_patches2(imagens, n_div, rgb=False): 
    # condicao de parada da recursao    
    if (n_div == 0):
        return (imagens);
        
    colecao = []
    for imagem in imagens:    
        if not(rgb):
            l, h = imagem.shape  # retorna largura e altura da imagem
        
            # calcula os valores medios
            h_m = int(math.ceil(h/2))             
            l_m = int(math.ceil(l/2))
            print ("l_m, h_m: ", str((l_m, h_m)))
            # divide a imagem passada em 4 patches        
            # patch 0,0            
            m = np.copy(imagem[:l_m,:h_m])                
            colecao.append(m)                    
            # patch 1,0            
            m = np.copy(imagem[:l_m,h_m:])                
            colecao.append(m)                    
            # patch 0,1        
            m = np.copy(imagem[l_m:,:h_m])                
            colecao.append(m)                    
            # patch 1,1
            m = np.copy(imagem[l_m:,h_m:])                
            colecao.append(m)                    
        else:
            l, h, _ = imagem.shape  # retorna largura e altura da imagem
            # calcula os valores medios
            h_m = int(math.ceil(h/2))             
            l_m = int(math.ceil(l/2))   
            #print ("l_m, h_m: ", str((l_m, h_m)))
            # divide a imagem passada em 4 patches        
            # patch 0,0            
            m = np.copy(imagem[:l_m:,:h_m:])                
            colecao.append(m)                    
            # patch 1,0            
            m = np.copy(imagem[:l_m:,h_m::])                
            colecao.append(m)                    
            # patch 0,1        
            m = np.copy(imagem[l_m::,:h_m:])                
            colecao.append(m)                    
            # patch 1,1
            m = np.copy(imagem[l_m::,h_m::])                
            colecao.append(m)
            
    n_div -= 1
    return (cria_patches(colecao, n_div, rgb)) 

'''
Utilizado para testar a funcao de criacao de patches
'''
def exibe_cria_patches(lista_imagens):
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
        patches = [img]
        for n_div in range(1,2):        
            patches = cria_patches(patches, 1, rgb=True)
            print("Total de patches: %f", len(patches))
            print("Tamanho do patch: %i", patches[0].shape)
                    
            y = int(math.sqrt(4**n_div))
            x = y 
            
            #fig,axes = plt.subplots(x,y, figsize=(32,32))          
            fig,axes = plt.subplots(x,y) 
            for i, ax in enumerate(axes.ravel()): 
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                im = ax.imshow(patches[i],'gray')         
            plt.show()    

def classifica(base, base_tr, id_clf, n, svm_c=SVM_C, svm_g=SVM_G):
    inicio = time()
    
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
    
    # Carrega a base de treinamento   
    print("Carregando base de treinamento...")     
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    #inicio = exibe_tempo(inicio)     
    
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
        print("Extraindo patches da imagem " + imagem)        
        atrib_vl, rotulos_vl = extrai([imagem], n)
        
        #inicio = exibe_tempo(inicio)         
        
        if len(atrib_vl) > 0:
            # Para o classificador QDA deve-se usar uma matriz densa
            if id_clf == "qda":    
                atrib_vl = np.asarray(atrib_vl)
            
            print("Predizendo classe da imagem")
            # predicao do classificador para o conjunto de patches        
            ls_preds = clf.predict(atrib_vl) 
            #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
            ls_preds = np.asarray(ls_preds, dtype='int32')                        
            conta = np.bincount(ls_preds)
            #print('conta ' + str(conta))
            rotulos_pred.append(np.argmax(conta))
            
            #inicio = exibe_tempo(inicio) 
    
    
    return (rotulos_ts, rotulos_pred)

def classifica2(base, atrib_tr, rotulos_tr, id_clf, n, svm_c=SVM_C, svm_g=SVM_G):
    inicio = time()
    
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
    
    # Carrega a base de treinamento   
    print("Carregando base de treinamento...")     
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    #inicio = exibe_tempo(inicio)     
    
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
        print("Extraindo patches da imagem " + imagem)        
        atrib_vl, rotulos_vl = extrai([imagem], n)
        
        #inicio = exibe_tempo(inicio)         
        
        if len(atrib_vl) > 0:
            # Para o classificador QDA deve-se usar uma matriz densa
            if id_clf == "qda":    
                atrib_vl = np.asarray(atrib_vl)
            
            print("Predizendo classe da imagem")
            # predicao do classificador para o conjunto de patches        
            ls_preds = clf.predict(atrib_vl) 
            #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
            ls_preds = np.asarray(ls_preds, dtype='int32')                        
            conta = np.bincount(ls_preds)
            #print('conta ' + str(conta))
            rotulos_pred.append(np.argmax(conta))
            
    inicio = exibe_tempo(inicio, "CLASSIFICACAO para " + str(4**n) + " patches") 
    
    
    return (rotulos_ts, rotulos_pred)

   
'''
Para cada imagem da lista de imagens passada, divide a imagem em patches
e para cada patch aplica um extrator de descritor de textura 
'''
def extrai(lista_imgs, n_div, descarta=False):
    atributos = []    
    rotulos = []    
    ref = ex.patch_referencia()
    hist_ref = bp.histograma(bp.aplica_lbp(ref)) 
    descartados = []
    
    for arquivo in lista_imgs:        
        # recupera do nome do arquivo a sua classe 
        classe, _ = ex.classe_arquivo(arquivo)
                
        #converte para escala de cinza
        img = mh.imread(arquivo)            
        patches = cria_patches([img], n_div, rgb=True)
                        
        if (descarta):
            img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
            img_cinza = bp.limpa_imagem(img_cinza)                    
            patches_cinza = cria_patches([img_cinza], n_div)
        
        # calcula o histograma de cada um dos patches    
        for i, patch in enumerate(patches):
            if (descarta): 
                lbp_patch = bp.aplica_lbp(patches_cinza[i])
                hist = bp.histograma(lbp_patch)                  
                dist = bp.distancia_histograma(hist, hist_ref)  
                if (dist > LIMIAR_DIVERGENCIA): 
                    p_pftas = mh.features.pftas(patch)            
                    atributos.append(p_pftas)  
                    rotulos.append(ex.CLASSES[classe])
                else:
                    dist = str("Distância: " + '{:f}'.format(dist)) 
                    descartados.append((patch, dist))
            else:                
                    p_pftas = mh.features.pftas(patch)           
                    atributos.append(p_pftas)  
                    rotulos.append(ex.CLASSES[classe])                             
        
    return (atributos,rotulos)    
    
def extrai2(img, classe, n_div=1):
    atributos = []    
    rotulos = []    
    
        
    patches = cria_patches([img], n_div, rgb=True)
    
    # calcula o histograma de cada um dos patches    
    for i, patch in enumerate(patches):
        p_pftas = mh.features.pftas(patch)           
        atributos.append(p_pftas)  
        rotulos.append(ex.CLASSES[classe])                             

    return (atributos,rotulos, patches)
    
def exibe_tempo(inicio, desc=""):
    fim = round(time() - inicio, 3)
    print ("Tempo total de execução ["+desc+"]: " +str(fim))
    
    return(time())     
        
  
# Inicia contagem do tempo de execução
inicio1 = time()

diretorio="/home/willian/basesML/bases_cancer/fold1/treino/"
#diretorio="/home/willian/basesML/bases_cancer/min_treino2/"
#base_teste="/home/willian/basesML/bases_cancer/min_teste2/"
base_teste="/home/willian/basesML/bases_cancer/fold1/teste/"

lista_imagens = arq.busca_arquivos(diretorio, "*.png")
n_imgs_treino = len(lista_imagens)



#base_treino=""

tamanhos = []
taxas = []
matrizes = []
classfs = ["dt", "svm"]

total_n = 4

atributos = {}      # dicionario de atributos por qtd de patches
rotulos = {}        # dicionario de rotulos por qtd de patches
arqs_treino = {}    # dicionario de arquivos de treino por qtd de patches

for n in range(0,total_n):        
    atributos[n] = []    
    rotulos[n] = []     
    arq_saida = diretorio + "base_PFTAS_"+str(n)+"_divs.svm" 
    arqs_treino[n] = arq_saida

##  INICIO DO PROCESSO DE EXTRACAO DE ATRIBUTOS    
'''
for arq_imagem in lista_imagens:
    imagem = mh.imread(arq_imagem)
    classe, _ = ex.classe_arquivo(arq_imagem)
    
    # Extrai os atributos e gera os arquivos dos patches da base de treino
    atrs,rots,patches = extrai2(imagem, classe, 0)    
    atributos_img = atributos.get(0)
    atributos_img += atrs     
    rotulos_img = rotulos.get(0)
    rotulos_img += rots
    
    # Executa para todos os tamanhos de patches e acumula as listas de atributos
    # e os rotulos de cada um dos patches gerados
    # A vantagem desse metodo é realizar apenas uma leitura em disco e evitar 
    # excesso de overhead ao dividir a imagem em imagens menores (patches)
    # pois ja divide a imagem e extrai atributos para todos os tamanhos de patches
    for n in range(1, total_n): 
        novos_patches = []
                           
        for img in patches:         
            atrs, rots, p = extrai2(img, classe)  
            atributos_img = atributos.get(n)
            atributos_img += atrs     
            rotulos_img = rotulos.get(n)
            rotulos_img += rots
            
            novos_patches += p
        
        patches = novos_patches
    
for n in range(0, total_n):             
    dump_svmlight_file(atributos.get(n), rotulos.get(n), arqs_treino.get(n))
    atributos[n] = []    
    rotulos[n] = []
'''
inicio = exibe_tempo(inicio1, "EXTRACAO")

## FIM DO PROCESSO DE EXTRACAO DE ATRIBUTOS


## INICIO DO PROCESSO DE CLASSIFICACAO
# Processo de classificacao
print("Classificando a base de testes...")    

# Busca a lista dos arquivos da base de testes
lista_imagens = arq.busca_arquivos(base_teste, "*.png")
    
for n in range(0, total_n):        
    # Carrega a base de treinamento   
    base_tr = arqs_treino.get(n)
    atrib_tr = None
    rotulos_tr = None    
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
    
    # Classifica a base de testes
    for c in classfs:
        inicio = time()           
        print("Classificando para " + c + " com " + str(4**n) + " patches")
                
        r_tst,r_pred = classifica2(base_teste, atrib_tr, rotulos_tr, c, n)
        
        # cria as matrizes de confusao
        cm = confusion_matrix(r_tst, r_pred)
        print("Matriz de confusao: ")
        print (cm) 
        
        # exibe a taxa de classificacao
        r_pred = np.asarray(r_pred)
        r_tst = np.asarray(r_tst)
        taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
        print("Taxa de Classificação: %f " % (taxa_clf))     
        
        # armazena os resultados
        tamanhos.append(4**n)
        taxas.append(taxa_clf)

        matrizes.append(cm)    
        inicio = exibe_tempo(inicio, "CLASSIFICACAO")
    
    
## FIM DO PROCESSO DE CLASSIFICACAO

# exibe os dados obtidos
for t,tx in zip(tamanhos,taxas):
    print("Tamanho: " + str(t) + " Taxa Reconhecimento: " + str(tx))        
    
# plota grafico de resultados    
plt.plot(tamanhos, taxas)
plt.ylabel('taxa de reconhecimento')
plt.xlabel('qtd de patches')
plt.show()            

# Encerramento e contagem de tempo de execucao
print("ENCERRAMENTO DO PROGRAMA \n\n")
inicio = exibe_tempo(inicio1)      
    
 

            
'''
CLASSIFICACAO E EXTRACAO FUNCIONANDO, MAS AINDA LENTO
main():
    # Inicia contagem do tempo de execução
    inicio1 = time()
    
    diretorio="/home/willian/basesML/bases_cancer/min_treino/"
    base_teste="/home/willian/basesML/bases_cancer/min_teste/"
    
    lista_imagens = arq.busca_arquivos(diretorio, "*.png")
    
    
    #base_treino=""
    arqs_treino = []
    tamanhos = []
    taxas = []
    matrizes = []
    classfs = ["dt", "svm"]
    
    
    for n in range(0,4):  
        print("Extraindo atributos da base de treino para " + str(4**n) + " patches")
        # Extrai os atributos e gera os arquivos dos patches da base de treina
        atributos, rotulos, lista_imagens = extrai(lista_imagens, n)  
        arq_saida = diretorio + "base_PFTAS_"+str(2*n)+"x"+str(2*n)+".svm" 
        arqs_treino.append(arq_saida)    
        dump_svmlight_file(atributos, rotulos, arq_saida)
    
        inicio = exibe_tempo(inicio1) 
        
        print("Classificando a base de testes...")    
        # Classifica a base de testes
        for c in classfs:           
            print("Classificando para " + c + " com " + str(4**n) + " patches")
                    
            r_tst,r_pred = classifica(base_teste, arq_saida, c, n)
            
            inicio = exibe_tempo(inicio)
    
            # cria as matrizes de confusao
            cm = confusion_matrix(r_tst, r_pred)
            print("Matriz de confusao: ")
            print (cm) 
            
            # exibe a taxa de classificacao
            r_pred = np.asarray(r_pred)
            r_tst = np.asarray(r_tst)
            taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
            print("Taxa de Classificação: %f " % (taxa_clf))     
            
            # armazena os resultados
            tamanhos.append(4**n)
            taxas.append(taxa_clf)
            matrizes.append(cm)    
    
    
    # exibe os dados obtidos
    for t,tx in zip(tamanhos,taxas):
        print("Tamanho: " + str(t) + " Taxa Reconhecimento: " + str(tx))        
        
    # plota grafico de resultados    
    plt.plot(tamanhos, taxas)
    plt.ylabel('taxa de reconhecimento')
    plt.xlabel('qtd de patches')
    plt.show()            
    
    # Encerramento e contagem de tempo de execucao
    print("ENCERRAMENTO DO PROGRAMA \n\n")
    inicio = exibe_tempo(inicio1)   
'''

# Programa principal
