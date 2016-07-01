# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016

@author: antoniosousa

Testes de classificação
"""
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
import binarypattern as bp
import arquivos as arq
import numpy as np
import extrator as ex
import mahotas as mh
import math
import cv2
import matplotlib.pyplot as plt


LIMIAR_DIVERGENCIA=0.2
CLASSES = {'B':0, 'M':1}
SUBCLASSES = {'A':0, 'F':1, 'TA':2, 'PT':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}    
RAIO = 3
PONTOS = 24
TAM_PATCH = 64
N_DIV = 2

def cria_patches(imagens, n_div, rgb=False): 
    # condicao de parada da recursao    
    if (n_div == 0):
        return (imagens);
        
    colecao = []
    for imagem in imagens:    
        if not(rgb):
            l, h = imagem.shape  # retorna largura e altura da imagem
        else:
            l, h = imagem.shape[0]  # retorna largura e altura da imagem
        
        # calcula os valores medios
        h_m = h//2
        l_m = l//2
        
        # divide a imagem passada em 4 patches
        m = np.copy(imagem[:l_m,:h_m])                
        colecao.append(m)            
        m = np.copy(imagem[l_m+1:,:h_m])                
        colecao.append(m)            
        m = np.copy(imagem[:l_m,h_m+1:])                
        colecao.append(m)            
        m = np.copy(imagem[l_m+1:,h_m+1:])                
        colecao.append(m)            
        
    n_div -= 1
    return (cria_patches(colecao, n_div, rgb))    



def div_patches(img, num, rgb=False):
    img = np.asarray(img)     
    window_size = (tam,tam) if not(rgb) else (tam,tam,3)
    windows = sw.sliding_window_nd(img, window_size)  
    
    return (windows)


def pftas(patch):
   return (mh.features.pftas(patch)) 
   
   
'''
Para cada imagem da lista de imagens passada, divide a imagem em patches
e para cada patch aplica um extrator de descritor de textura 
'''
def extrai(lista_imgs, extrator, tam_patch=32, descarta=False):
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
        patches = bp.patches(img, TAM_PATCH, rgb=True)
                        
        if (descarta):
            img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
            img_cinza = bp.limpa_imagem(img_cinza)                    
            patches_cinza = bp.cria_patches(img_cinza, TAM_PATCH)
        
        # calcula o histograma de cada um dos patches    
        for i in range(patches.shape[0]):
            patch = patches[i]
            if (descarta): 
                lbp_patch = bp.aplica_lbp(patches_cinza[i])
                hist = bp.histograma(lbp_patch)                  
                dist = bp.distancia_histograma(hist, hist_ref)  
                if (dist > LIMIAR_DIVERGENCIA): 
                    p_pftas = extrator(patch)            
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe])
                else:
                    dist = str("Distância: " + '{:f}'.format(dist)) 
                    descartados.append((patch, dist))
            else:                
                    p_pftas = extrator(patch)           
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe])                             
        
    return (atributos,rotulos)    

diretorio="/home/willian/basesML/bases_cancer/min_treino_2/"
lista_imagens = arq.busca_arquivos(diretorio, "*.png")

base_teste=""
base_treino=""
#r_tst,r_pred = fusao_serie(base_teste, base_treino, ["", ""], metodo)

#converte para escala de cinza
for arquivo in lista_imagens:
    img = mh.imread(arquivo)            
    
    #patches = bp.patches(img, TAM_PATCH, rgb=True)
    img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
    #img_cinza = bp.limpa_imagem(img_cinza)                    
    #patches_cinza = bp.cria_patches(img_cinza, TAM_PATCH)        
    #patches_cinza = bp.patches(img_cinza, TAM_PATCH, rgb=False)        
    
    '''
    for p,pg in zip(patches,patches_cinza):
        plt.imshow(p)
        plt.show()
        plt.imshow(pg, 'gray')
        plt.show()
    ''' 
    # exibe a imagem original
    plt.imshow(img_cinza, 'gray')    
    n_div = 3
    patches = cria_patches([img_cinza], n_div)
    x = int(math.sqrt(4**n_div))
    y = x 
    #p = patches.ravel()
    fig,axes = plt.subplots(x,y, figsize=(16,16)) 
    #fig.subplots_adjust(hspace=.5) 
    
    for i, ax in enumerate(axes.ravel()):          
        im = ax.imshow(patches[i],'gray')         
    
    #plt.show()
    
        
        
        
    

        








'''
qt_desc = len(descartados)
x = int(math.sqrt(qt_desc) + 1)
y = int(qt_desc/x)
#fig,axes = plt.subplots(x,y, figsize=(16,16)) 
fig,axes = plt.subplots(x,y, figsize=(16,16)) 
fig.subplots_adjust(hspace=.5) 
conta = 0 
for i, ax in enumerate(axes.ravel()):  
    if i < qt_desc:
        conta += 1
        im = ax.imshow(descartados[i][0],'gray') 
        ax.set_title(descartados[i][1])
        if (conta % 10) == 0:
            plt.savefig("descartados_" + str(conta) + ".png")
            fig,axes = plt.subplots(10,10, figsize=(16,16))
            fig.subplots_adjust(hspace=.5)
           
#plt.savefig("descartados.png")
#plt.show()
'''





'''
diretorio="/home/willian/basesML/bases_cancer/treino/"
extratores=['lbp', 'glcm', 'pftas']
padrao='*.png'

lista_imagens = arq.busca_arquivos(diretorio, padrao)



for extrator in extratores:

    if (extrator == 'lbp'):
        caracteristicas, rotulos = ex.extrai_lbp(lista_imagens)
        arq_saida = diretorio + "base_LBP.svm"          
    elif (extrator == 'pftas'):
        caracteristicas, rotulos = ex.extrai_pftas(lista_imagens)  
        arq_saida = diretorio + "base_PFTAS.svm"
    elif (extrator == 'glcm'):
        caracteristicas, rotulos = ex.extrai_haralick(lista_imagens) 
        arq_saida = diretorio + "base_GLCM.svm"                 
        
    dump_svmlight_file(caracteristicas, rotulos, arq_saida)        
'''   

"""
TAS and PFTAS: TAS (Threshold Adjacency Statistics) were presented by Hamilton et al. in 2007 [1]. Originally, TAS was proposed as a ”simple and fast morphological measure for distinguishing sub-cellular localization” [20, p.8]. One variation is do not use hardcoded parameters in threshold value, but compute it using an algorithm such as Otsu’s method. This alternative approach, named PFTAS (Parameter Free-TAS), was first published by Coelho et al. in 2010 [2].

  [1] N. A. Hamilton, R. S. Pantelic, K. Hanson, and R. D. Teasdale, “Fast automated cell phenotype image classification,” BMC Bioinformatics, vol. 8, 2007. [Online]. Available: http://www.biomedcentral.com/
1471-2105/8/110

  [2] L. P. Coelho, A. Ahmed, A. Arnold, J. Kangas, A.-S. Sheikh, E. P. Xing, W. W. Cohen, and R. F. Murphy, “Structured literature image finder: extracting information from text and images in biomedical literature,” in
Linking Literature, Information, and Knowledge for Biology, ser. Lecture Notes in Computer Science, C. Blaschke and H. Shatkay, Eds. Springer Berlin Heidelberg, 2010, vol. 6004, pp. 23–32.


"""


'''
def mTAS(img1):

    #img_g = img[:, :, 0]  # grayscale. It is good enough?

    # convert to grayscale using OpenCV
    #img_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img_g = img1
    thresh = 30
    margin = 30
    total = np.sum(img_g > thresh)
    mu = ((img_g > thresh)*img_g).sum() / (total + 1e-8)

    print ('total:%f' % total)
    print ('mu:%f' % mu)
    print ('mu+30:%f' % (mu+30))

    ret, mu1 = cv2.threshold(img_g, (mu-margin), (mu+margin), cv2.THRESH_BINARY)
    ret, mu2 = cv2.threshold(img_g, mu-margin, 255, cv2.THRESH_BINARY)
    ret, mu3 = cv2.threshold(img_g, mu, 255, cv2.THRESH_BINARY)

    mu1 = (img_g > mu - margin) * (img_g < mu + margin)
    mu2 = img_g > mu - margin
    mu3 = img_g > mu

    #'original converted to grayscale'
    titles = ['original image', r'$\mu-30$ to $\mu+30$', r'$\mu-30$ to $255$', r'$\mu$ to $255$']
    images = [img_g, mu1, mu2, mu3]

    #plt.imshow(img_g, 'gray')
    #plt.show()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02)

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    #PFTAS

    T = mh.otsu(img_g)
    pixels = img_g[img_g > T].ravel()  #  pixels becames a flattened array

    if len(pixels) == 0:
        std = 0
    else:
        std = pixels.std()

    thresh = T
    margin = std


    total = np.sum(img_g > thresh)
    mu = ((img_g > thresh)*img_g).sum() / (total + 1e-8)

    print ('total:%f' % total)
    print ('mu:%f' % mu)
    print ('sigma:%f' % std)

    ret, mu1 = cv2.threshold(img_g, (mu-margin), (mu+margin), cv2.THRESH_BINARY)
    ret, mu2 = cv2.threshold(img_g, mu-margin, 255, cv2.THRESH_BINARY)
    ret, mu3 = cv2.threshold(img_g, mu, 255, cv2.THRESH_BINARY)

    mu1 = (img_g > mu - margin) * (img_g < mu + margin)
    mu2 = img_g > mu - margin
    mu3 = img_g > mu

    # 'original converted to grayscale'
    titles = ['original image', r'$\mu-\sigma$ to $\mu+\sigma$', r'$\mu-\sigma$ to $255$', r'$\mu$ to $255$']
    images = [img_g, mu1, mu2, mu3]

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

f1 = lista_imagens[0]
img = mh.imread(f1)
mTAS(img) 
'''
    # patch de referencia            
    #if len(ref)==0:
    #    patch_branco = np.full([TAM_PATCH,TAM_PATCH], 255, dtype=np.uint8)    
    #    ref = bp.histograma(bp.aplica_lbp(patch_branco)) 
    #        
