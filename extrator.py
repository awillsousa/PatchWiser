# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:17:46 2016

@author: antoniosousa
"""

import numpy as np
import mahotas as mh
import binarypattern as bp
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern 
import cv2 
    
LIMIAR_DIVERGENCIA=0.25 
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
        

'''
Intera sobre uma lista de imagens, dividindo cada imagem em um conjunto de patches
de tamanho 64x64. Para cada um desses patches, obtem a sua matriz lbp (24,3) e 
calcula o histograma de cada um deles. 
O flag usa_descarte indica se os patches devem ser avaliados antes de serem utilizados
e serem descartados ou não dependendo do retorno da funcao que avalia o potencial de 
informacao que o patch pode fornecer
'''
def extrai_lbp(lista_imgs, usa_descarte=False):
    
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
            h = np.asarray(hist)            
            h = h.T.flatten()            
            
            if (usa_descarte):
                dist = bp.distancia_histograma(hist, hist_ref)  
                if (dist > LIMIAR_DIVERGENCIA):
                    histogramas.append(h)  
                    rotulos.append(CLASSES[classe])                     
                else:
                    dist = str("Distância: " + '{:f}'.format(dist)) 
                    descartados.append((patch, dist))
            else:                
                    histogramas.append(h)  
                    rotulos.append(CLASSES[classe])                            
    
    return (histogramas,rotulos)

'''
Itera sobre uma lista de imagens, dividindo cada imagem em um conjunto de patches
de tamanho 64x64. Para cada um desses patches, obtem um vetor de caracteristicas 
usando pftas. 
O flag usa_descarte indica se os patches devem ser avaliados antes de serem utilizados
e serem descartados ou não dependendo do retorno da funcao que avalia o potencial de 
informacao que o patch pode fornecer
'''
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
                    p_pftas = mh.features.pftas(patch)            
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe])
                else:
                    dist = str("Distância: " + '{:f}'.format(dist)) 
                    descartados.append((patch, dist))
            else:                
                    p_pftas = mh.features.pftas(patch)           
                    atributos.append(p_pftas)  
                    rotulos.append(CLASSES[classe])                             
        
    return (atributos,rotulos)
    
def extrai_haralick(lista_imgs, usa_descarte=False):
    atributos = []    
    rotulos = []    
    ref = patch_referencia() 
    hist_ref = bp.histograma(bp.aplica_lbp(ref)) 
    descartados = [] 
    
    for arquivo in lista_imgs:        
        # recupera do nome do arquivo a sua classe 
        classe, _ = classe_arquivo(arquivo) 
                
        #converte para escala de cinza 
        img_cinza = mh.imread(arquivo, as_grey=True)            
        
        if (usa_descarte):
            img_cinza = limpa_imagem(img_cinza)        
        patches = bp.cria_patches(img_cinza, TAM_PATCH)
        
        # calcula o histograma de cada um dos patches    
        for patch in patches:
            if (usa_descarte): 
                lbp_patch = bp.aplica_lbp(patch)
                hist = bp.histograma(lbp_patch)                  
                if (patch_valido(hist, hist_ref)):
                    glcm = mh.features.haralick(np.asarray(patch, dtype=np.uint8))
                    glcm = np.asarray(glcm).flatten()           
                    atributos.append(glcm)  
                    rotulos.append(CLASSES[classe])  
            else:                     
                    glcm = mh.features.haralick(np.asarray(patch, dtype=np.uint8))
                    glcm = np.asarray(glcm).flatten()           
                    atributos.append(glcm)  
                    rotulos.append(CLASSES[classe])                           
        
    return (atributos,rotulos)

