# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016

@author: antoniosousa

Realiza a extraçao de caracteristicas e armazena num arquivo em formato SVM Light
"""
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
import binarypattern as bp
import arquivos as arq
import numpy as np
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


def limpa_imagem(img_cinza):
    #binariza a imagem em escala de cinza
    img_bin_cinza = np.where(img_cinza < np.mean(img_cinza), 0, 255)
    
    # aplica lbp sobre a imagem em escala de cinza
    # lbp foi aplicado para evitar perda de informacao em regioes
    # proximas as regioes escuras (provaveis celulas)
    lbp_img = local_binary_pattern(img_cinza, 24, 3, method='uniform')
    
    # aplica efeito de blurring sobre a imagem resultante do lbp 
    blur_img = gaussian(lbp_img,sigma=6)    
    img_bin_blur = np.where(blur_img < np.mean(blur_img), 0, 255)
     
    # junta as duas regiões definidas pela binarizacao da imagem em escala
    # de cinza e a binarizacao do blurring    
    mascara = np.copy(img_bin_cinza)    
    for (a,b), valor in np.ndenumerate(img_bin_blur):
        if valor == 0:        
            mascara[a][b] = 0 
            
    # aplica a mascara obtida sobre a imagem original (em escala de cinza)
    # para delimitar melhor as regiões que não fornecerao informacoes (regioes
    # totalmente brancas)
    img_limpa = np.copy(img_cinza)
    for (a,b), valor in np.ndenumerate(mascara):
        if valor == 255:
            img_limpa[a][b] = 255

    return (img_limpa)

def classe_arquivo(nome_arquivo):
    info_arquivo =  str(nome_arquivo[nome_arquivo.rfind("/")+1:]).split('_')        
    classe = info_arquivo[1]
    subclasse = info_arquivo[2].split('-')[0]
    
    return (classe,subclasse)

def patch_referencia():
    patch_ref = np.full([TAM_PATCH,TAM_PATCH], 255, dtype=np.uint8)    
        
    return (patch_ref)

def patch_valido(hist, hist_ref):    
    r = bp.distancia_histograma(hist, hist_ref)    
    if (r > LIMIAR_DIVERGENCIA):
        return (True) 
    
    return(False)
    
    
def extrai_lbp(lista_imgs, usa_descarte=True):
    
    histogramas = []    
    rotulos = []    
    ref = patch_referencia()
    hist_ref = bp.histograma(bp.aplica_lbp(ref))
    descartados = []
    for arquivo in lista_imgs:
        print("Extraindo arquivo " + arquivo)        
        # recupera do nome do arquivo a sua classe 
        classe, _ = classe_arquivo(arquivo)
                
        #converte para escala de cinza
        img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
                     
        if (usa_descarte):
            img_cinza = limpa_imagem(img_cinza)
                    
        patches_cinza = bp.cria_patches(img_cinza, TAM_PATCH)        
       
        # calcula o histograma de cada um dos patches    
        for patch in patches_cinza:                        
            lbp_patch = bp.aplica_lbp(patch)
            hist = bp.histograma(lbp_patch)
            
            if (usa_descarte): 
                dist = bp.distancia_histograma(hist, hist_ref)                
                #print("Distância: " + str(dist))
                if (dist > 0.25):                                        
                    h = np.asarray(hist)            
                    h = h.T.flatten()            
                    histogramas.append(h)  
                    rotulos.append(CLASSES[classe]) 
                else:
                    if (dist > 0.0):
                        dist = str("Distância: " + '{:f}'.format(dist)) 
                        descartados.append((patch, dist))          
            else:
                    h = np.asarray(hist)            
                    h = h.T.flatten()                            
                    histogramas.append(h)  
                    rotulos.append(CLASSES[classe])         
            
    print("Total de descartados " + str(len(descartados)))        
    return (histogramas,rotulos, descartados)

def extrai_pftas(lista_imgs, usa_descarte=False):
    atributos = []    
    rotulos = []    
    ref = patch_referencia()
    hist_ref = bp.histograma(bp.aplica_lbp(ref)) 
    descartados = []
    
    for arquivo in lista_imgs:        
        # recupera do nome do arquivo a sua classe 
        classe, _ = classe_arquivo(arquivo)
                
        #converte para escala de cinza
        img = mh.imread(arquivo)            
        patches = bp.patches(img, TAM_PATCH, rgb=True)

        # Exibe a imagem atual
        #plt.imshow(img)
        #plt.show()        
                
        if (usa_descarte):
            img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)    
            img_cinza = limpa_imagem(img_cinza)                    
            patches_cinza = bp.cria_patches(img_cinza, TAM_PATCH)
        
        # calcula o histograma de cada um dos patches    
        for i in range(patches.shape[0]):
            patch = patches[i]
            if (usa_descarte): 
                lbp_patch = bp.aplica_lbp(patches_cinza[i])                
                hist = bp.histograma(lbp_patch)  
                
                dist = bp.distancia_histograma(hist, hist_ref)  
                if (dist > LIMIAR_DIVERGENCIA):                   
                    #plt.imshow(patch)
                    #plt.show() 
                    p_pftas = mh.features.pftas(patch)            
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe]) 
                else:
                    #plt.imshow(patch)
                    #plt.show() 
                    dist = str("Distância: " + '{:f}'.format(dist)) 
                    descartados.append((patch, dist))
            else:                
                    p_pftas = mh.features.pftas(patch)           
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe])                             
        
    return (atributos,rotulos,descartados)

diretorio="/home/willian/basesML/bases_cancer/treino/"

lista_imagens = arq.busca_arquivos(diretorio, "*.png")
atributos, rotulos, descartados = extrai_pftas(lista_imagens, True)


qt_desc = len(descartados)
x = int(math.sqrt(qt_desc) + 1)
y = int(qt_desc/x)

fig,axes = plt.subplots(4,4, figsize=(16,16)) 
fig.subplots_adjust(hspace=.5)     
conta = 0 
ax = axes.ravel() 
#while (conta < qt_desc+1):    
for patch,texto in descartados:
    
   im = ax[conta % 16].imshow(patch,'gray') 
   ax[conta % 16].set_title(texto) 
    
   if (conta % 16) == 0 and (conta != 0):                 
                plt.savefig("descartados_" + str(conta) + ".png")         
                fig,axes = plt.subplots(4,4, figsize=(16,16)) 
                fig.subplots_adjust(hspace=.5)   
                ax = axes.ravel() 
   conta += 1 

plt.savefig("descartados_" + str(conta) + ".png")           
#plt.savefig("descartados.png") 
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
