from src.vem_svd import vem_svd
from src.optim_icl import optim_icl
from src.power import pvalues_edges, pvalues_nodes
import numpy as np
import scipy.sparse as sp


if __name__ == "__main__":
    
    # nb_node: int = 10 # dim des matrices d'adjacences 
    # X_sum: np.array = np.random.randint(600, size=(nb_node, nb_node)) # somme des matrices d'adjacences
    # K:int = 10 # nombre de cluster (param utilisateur)
    # G:int = 5 # len() # len(matrices d'adjacences)
    
    # res = vem_svd(X_sum, K, nb_node, G)
    # print(res)
    
    
    nb_node:int = 10
    X:list[np.sp] = [
        sp.sparse(np.random.randint(600, size=(nb_node, nb_node)))
    ]
    G:int = len(X)
    list_K:list[int] = [5,6,7]
    
    res = optim_icl(X,nb_node,G,list_K)
    print(res)