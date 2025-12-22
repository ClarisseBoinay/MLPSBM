import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

from .vem_svd import vem_svd
from .critere import icl


# essais choisi avec le critère icl
def optim_icl(sp_X, nb_node, G, list_K):
    sum_X = sum(sp_X)
    list_icl = []
    list_connectivite = []
    list_critere_variationnel = []

    sum_log_fact_ij = np.zeros(nb_node)
    sum_log_fact_ji = np.zeros(nb_node)
    
    for un_X in sp_X:
        coo = un_X.tocoo()
        log_fact_values = gammaln(coo.data + 1)
    
        # Accumulate sum by row (axis=1) and column (after transpose)
        np.add.at(sum_log_fact_ij, coo.row, log_fact_values)
        np.add.at(sum_log_fact_ji, coo.col, log_fact_values)
    
    sum_log_fact = np.array(sum_log_fact_ij + sum_log_fact_ji).reshape(nb_node,1)
    X_log_fact = np.sum(sum_log_fact_ij)
    
    X_log_fact = np.sum(sum_log_fact_ij)

    for K_test in list_K:
        print("K")
        print(K_test)
        log_tau, log_vect_pi, log_mat_lambda, mon_J, proba_complete = vem_svd(sp_X,sum_X, K_test, nb_node, G, X_log_fact,sum_log_fact)
        print("monJ")
        print(mon_J)
        list_critere_variationnel.append(mon_J)
        monicl = icl(proba_complete, K_test, nb_node, G)
        print("monicl")
        print(monicl)
        list_icl.append(monicl)
        list_connectivite.append([log_tau, log_vect_pi, log_mat_lambda])
    monindex = list_icl.index(max(list_icl))
    K_star = list_K[monindex]
    l = list_connectivite[monindex]
    log_tau, log_vect_pi, log_mat_lambda = l[0], l[1], l[2]

    return (K_star, log_tau, log_vect_pi, log_mat_lambda, list_critere_variationnel, list_icl, list_connectivite)
