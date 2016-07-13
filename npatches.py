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

def divide4(x,y,x0=0,y0=0,k=1):
    if (k == 0):
        return x,y
        
    # calcula os valores medios
    y_m = int(math.floor((y - y0)/2))             
    x_m = int(math.floor((x - x0)/2))
    k -= 1
    
    return (divide4(x_m, y_m, x0, y0, k))    

def cria_patches2(imagens, n_div, rgb=False): 
    for imagem in imagens:            
        if (rgb):
            l,h,_ = imagem.shape
        else:    
            l, h = imagem.shape    
            
        tamX, tamY = divide4(l,h,0,0,n_div)    
        imagem = np.asarray(imagem)     
        window_size = (tamX,tamY) if not(rgb) else (tamX,tamY,3)
        windows = sw.sliding_window_nd(imagem, window_size)  
        print (windows.shape)
    return (windows)

def cria_patches3(imagem, lado, rgb=False):    
    from sklearn.feature_extraction import image
     
    #l, h,_ = imagem.shape    
    imagem = np.asarray(imagem)     
    #windows = image.extract_patches_2d(imagem, (lado,lado), max_patches=1000, random_state=0)     
    windows = image.PatchExtractor((lado, lado)).transform(imagem)    
    '''    
    imagem = np.asarray(imagem)     
    window_size = (lado,lado) if not(rgb) else (lado,lado,3)   
    windows = sw.sliding_window_nd(imagem, window_size)  
    '''
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
Executa a classificacao de uma base de imagens 
'''
def classifica(base, atrib_tr, rotulos_tr, id_clf, n, svm_c=SVM_C, svm_g=SVM_G):
    
    # Extrai os atributos da base passada
    lista_imagens = arq.busca_arquivos(base, "*.png")
        
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
        log("Extraindo patches da imagem " + imagem)        
        atrib_vl, rotulos_vl = extrai([imagem], n)
        
        
        if len(atrib_vl) > 0:
            # Para o classificador QDA deve-se usar uma matriz densa
            if id_clf == "qda":    
                atrib_vl = np.asarray(atrib_vl)
            
            log("Predizendo classe da imagem")
            # predicao do classificador para o conjunto de patches        
            ls_preds = clf.predict(atrib_vl) 
            #ls_preds = np.asarray(ls_preds, dtype='uint32')                        
            ls_preds = np.asarray(ls_preds, dtype='int32')                        
            conta = np.bincount(ls_preds)
            log('Contagem classes patches: ' + str(conta))
            cl = np.argmax(conta)
            rotulos_pred.append(cl)
            log ('Classe: ' + str(cl))
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
        patches = cria_patches2([img], n_div, rgb=True)
        #patches = cria_patches2([img], n_div, rgb=True)
                        
        if (descarta):
            img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
            img_cinza = bp.limpa_imagem(img_cinza)                    
            patches_cinza = cria_patches2([img_cinza], n_div)
        
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
    
def extrai_pftas_patches(img, classe, n_div=1):
    atributos = []    
    rotulos = []
        
    patches = cria_patches([img], n_div, rgb=True)
    
    # calcula o histograma de cada um dos patches    
    for i, patch in enumerate(patches):
        p_pftas = mh.features.pftas(patch)           
        atributos.append(p_pftas)  
        rotulos.append(ex.CLASSES[classe])                             

    return (atributos,rotulos, patches)

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

def extrai_patches_imgs_n(lista_imagens, diretorio, n=5):
        
    atributos = []    
    rotulos = []     
        
    arqs_treino = gera_arqs_treino(diretorio, n)    
    ##  INICIO DO PROCESSO DE EXTRACAO DE ATRIBUTOS    
    
    for arq_imagem in lista_imagens:
        imagem = mh.imread(arq_imagem)
        classe, _ = ex.classe_arquivo(arq_imagem)
        
        # Extrai os atributos e gera os arquivos dos patches da base de treino
        atrs,rots,patches = extrai_pftas_patches(imagem, classe, n)                            
        atributos += atrs
        rotulos += rots
    
    dump_svmlight_file(atributos, rotulos, arqs_treino.get(n))
        


'''
Exibe o tempo passado a partir do inicio passado e retorna o tempo atual
em uma nova variável
'''    
def exibe_tempo(inicio, desc=""):
    fim = round(time() - inicio, 3)
    print ("Tempo total de execução ["+desc+"]: " +str(fim))
    
    return(time())     
  
'''
Verifica se uma opção passada existe na lista de argumentos do parser
'''  
def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False 

'''
Extrai atributos e gera os arquivos da base de treino passado utilizando
o metodo passado
'''
def executa_extracao(base_treino, metodo, n_divs=6):
    inicio = time()    
    
    lista_imgs_tr = arq.busca_arquivos(base_treino, "*.png")
    n_imgs_treino = len(lista_imgs_tr)
    extrai_patches_imgs(lista_imgs_tr, base_treino, n_divs)
    
    log("Extraidos atributos da base " + base_treino + " utilizando " + metodo + "\n para " + str(n_imgs_treino) + "imagens") 
  
    # Exibe o tempo de execução    
    log(str(time()-inicio) + "EXTRAÇÃO") 
    
def executa_extracao_n(base_treino, metodo, n=1):
    inicio = time()    
    
    lista_imgs_tr = arq.busca_arquivos(base_treino, "*.png")
    n_imgs_treino = len(lista_imgs_tr)
    extrai_patches_imgs_n(lista_imgs_tr, base_treino, n)
    
    log("Extraidos atributos da base " + base_treino + " utilizando " + metodo + "\n para " + str(n_imgs_treino) + "imagens") 
  
    # Exibe o tempo de execução    
    log(str(time()-inicio) + "EXTRAÇÃO")     
    
def executa_classificacao(base_teste, base_treino, total_n=5):
    inicio1 = time()    
    
    tamanhos = []
    taxas = []
    matrizes = []
    tempos = []
    classfs = ["svm"] #"dt",
    
    # Dicionario de arquivos de treino por qtd de patches    
    arqs_treino = gera_arqs_treino(base_treino, total_n)        
    for n in range(0, total_n):        
        try: 
            # Carrega a base de treinamento   
            base_tr = arqs_treino.get(n) 
            atrib_tr = None 
            rotulos_tr = None             
            atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=np.float32) 
            
            '''
            t = int(atrib_tr.shape[0])
            t = t//10            
            atrib_tr = atrib_tr[:t,:] 
            rotulos_tr = rotulos_tr[:t] 
            '''
            # Classifica a base de testes
            for c in classfs:
                inicio = time()           
                log("Classificando para " + c + " com " + str(4**n) + " patches")
                        
                r_tst,r_pred = classifica(base_teste, atrib_tr, rotulos_tr, c, n)
                
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
                tempos.append(time()-inicio) 
                inicio = exibe_tempo(inicio, "CLASSIFICACAO") 
        except FileNotFoundError as fnf:
            log("I/O error({0}): {1}".format(fnf.errno, fnf.strerror))
        except TypeError as te:                        
            log("Type Error: {1}".format(te.args))
        except:
            log("ERRO ou PROBLEMA desconhecido no processo de classificação.") 
            print ("Unexpected error:", sys.exc_info()[0])
            raise           
    # Exibe o tempo de execução    
    log("Tempo: " + str(time()-inicio1) + "CLASSIFICACAO para " + str(4**n) + " patches")     
    
    ## FIM DO PROCESSO DE CLASSIFICACAO
    
    # exibe os dados obtidos
    for t,tx,mc in zip(tamanhos,taxas,matrizes):
        log("\nTamanho: " + str(t) + " Taxa Reconhecimento: " + str(tx))        
        log("Matriz de Confusão: \n" + str(mc))        
        
    # plota grafico de resultados [reconhecimento vs tam. patch]
    arq_grafico = base_treino + 'base_PFTAS_'+str(n)+'_divs.pdf'
    plota_grafico(tamanhos, taxas, arq_grafico, tituloX="Num. Patches", tituloY="Tx. Reconhecimento")
    
    arq_grafico_tempo = base_treino + 'Tempos_'+str(n)+'_divs.pdf'
    plota_grafico(tamanhos, tempos, arq_grafico_tempo, tituloX="Num. Patches", tituloY="Tempo")

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
    classfs = ["svm"] #"dt", 
    
    try:
        # Carrega a base de treinamento         
        base_tr = base_treino + "base_PFTAS_"+str(n)+"_divs.svm"        
        atrib_tr = None
        rotulos_tr = None    
        atrib_tr, rotulos_tr = load_svmlight_file(base_tr)
        
        # Classifica a base de testes
        for c in classfs:
            inicio = time()           
            log("Classificando para " + c + " com " + str(4**n) + " patches")
                    
            r_tst,r_pred = classifica(base_teste, atrib_tr, rotulos_tr, c, n)
            
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
    arq_grafico = base_treino + 'base_PFTAS_'+str(n)+'_divs.pdf'
    plota_grafico(tamanhos, taxas, arq_grafico, tituloX="Num. Patches", tituloY="Tx. Reconhecimento")
    
    arq_grafico_tempo = base_treino + 'Tempos_'+str(n)+'_divs.pdf'
    plota_grafico(tamanhos, tempos, arq_grafico_tempo, tituloX="Num. Patches", tituloY="Tempo")


def trata_excecao(excecao):    
    log("I/O error({0}): {1}".format(excecao.errno, excecao.strerror))    
    raise    

'''
Executa classificacao da base de testes utilizando uma base de treino
de imagens que foram divididas em 4**n patches
'''    
def executa_classificacao_ns(base_teste, base_treino, n=1):
    inicio1 = time()    
    
    tamanhos = []
    taxas = []
    matrizes = []
    tempos = []
    classfs = ["svm"] #"dt", 
    
    try:
        # Carrega a base de treinamento   
        base_tr = base_treino + "base_PFTAS_"+str(n)+"_divs.svm"
        atrib_tr = None
        rotulos_tr = None    
        atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=np.float32)
        
        # Classifica a base de testes
        for c in classfs:
            inicio = time() 
            log("Classificando para " + c + " com " + str(4**n) + " patches")
                    
            r_tst,r_pred = classifica(base_teste, atrib_tr, rotulos_tr, c, n)
            
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
            tempos.append(time()-inicio) 
            inicio = exibe_tempo(inicio, "CLASSIFICACAO") 
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
    arq_grafico = base_treino + 'base_PFTAS_'+str(n)+'_divs.pdf'
    plota_grafico(tamanhos, taxas, arq_grafico, tituloX="Num. Patches", tituloY="Tx. Reconhecimento")

###############################################################################

def plota_grafico(dadosX, dadosY, arquivo="grafico.pdf", titulo="", tituloX="X", tituloY="Y", ):
    # plota grafico de resultados [reconhecimento vs tam. patch]
    plt.plot(dadosX, dadosY)
    
    # anota os pontos de classificacao
    for x,y in zip(dadosX,dadosY):
        #plt.annotate(r'$('+str(x)+","+str(round(y,2))+')$',
        #plt.annotate(r'$('+str(x)+","+str(round(y,2))+')$',
        #         xy=(x,y), xycoords='data',
        #         xytext=(+10, +30), textcoords='offset points', fontsize=10,
        #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1"))

        plt.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
        plt.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
        plt.scatter([x,],[y,], 50, color ='red')
        
    plt.ylabel(tituloY)
    plt.xlabel(tituloX)
    if (titulo == ""):
        titulo = tituloX + " vs " + tituloY
    plt.title(titulo)
    
    # set axis limits
    plt.xlim(0.0, len(dadosX)**4)
    max_y = max(dadosY)
    if max_y < 100:
        max_y = 100
        
    plt.ylim(0.0, max_y)
    #plt.savefig(arquivo, bbox_inches='tight')
    plt.savefig(arquivo)
    plt.clf()
       

###############################################################################

def main():
    inicio = time()
    
    parser = OptionParser()
    parser.add_option("-T", "--base-treino", dest="base_treino",
                      help="Localização do arquivo da base de treino")
    parser.add_option("-t", "--base-teste", dest="base_teste",
                      help="Localização do arquivo da base de teste")     
    parser.add_option("-C", "--clf", dest="opt_clf",
                      default='dt',
                      help="Lista de classificadores a serem utilizados. [dt, qda, svm, knn]")
    parser.add_option("-m", "--metodo", dest="opt_metodo",
                      default='pftas',
                      help="Lista de metodos de extração de atributos. [pftas, lbp, glcm]")                                
    parser.add_option("-X", "--extrator", dest="opt_ext",
                      default=False,
                      help="Indica se será realizada extração na base de treino.")                                  
    parser.add_option("-d", "--divs", dest="opt_hqtree",
                      #default=4,
                      help="Altura da quadtree de patches geradas.")    
    parser.add_option("-n", "--n-div", dest="opt_n",
                      #default=0,
                      help="Nível da quadtree de patches a ser utilizado.")                             
    
    (options, args) = parser.parse_args()
    
    # verifica se a base de treino e de teste passadas existem
    if existe_opt(parser, "base_treino") and (not (path.isdir(options.base_treino))):
       sys.exit("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")    
    if existe_opt(parser, "base_teste") and (not (path.isdir(options.base_teste))):
       sys.exit("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")  
    
    # Se for executar extracao
    if existe_opt(parser, "opt_ext"):
       # metodo a ser utilizado        
       if not(existe_opt(parser, "opt_metodo")):            
            sys.exit("Erro: Metodo de extração ausente.")
            
       executa_extracao(options.base_treino, options.opt_metodo)
               
    # Se for executar classificacao    
    if existe_opt(parser, "opt_clf"):
        if existe_opt(parser, "opt_n"):
            try: 
                n_div = int(options.opt_n)
            except:
                sys.exit("Erro: Valor inválido para o número de divisões das imagens.")
        
            if (n_div > 0):
                executa_classificacao_n(options.base_teste, options.base_treino, n_div)
            else:
                sys.exit("Erro: Valor inválido para o número de divisões das imagens.")
        elif existe_opt(parser, "opt_hqtree"):
            hqtree = int(options.opt_hqtree)
            executa_classificacao(options.base_teste, options.base_treino, hqtree)
            #exibe_cria_patches(options.base_teste, hqtree)
        else:
            sys.exit("Erro: É preciso especificar um nível ou profundidade da árvore de patches geradas.")


    # Exibe o tempo de execução    
    log("Tempo: " + str(time()-inicio) + " de EXECUÇÃO DO PROGRAMA")     
   
   # Encerramento e contagem de tempo de execucao 
    print("ENCERRAMENTO DO PROGRAMA \n\n")     
    logfile.close() 
    
# Programa principal 
if __name__ == "__main__":	
	main()