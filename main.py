from src.optim_icl import optim_icl
from src.power import pvalues_edges, pvalues_nodes
from src.ll_onegraph import ll_onegraph
import numpy as np
import scipy.sparse as sp
import pandas as pd


if __name__ == "__main__":
    
    # nb_node: int = 10 # dim des matrices d'adjacences 
    # X_sum: np.array = np.random.randint(600, size=(nb_node, nb_node)) # somme des matrices d'adjacences
    # K:int = 10 # nombre de cluster (param utilisateur)
    # G:int = 5 # len() # len(matrices d'adjacences)
    
    # res = vem_svd(X_sum, K, nb_node, G)
    # print(res)
    
    
    nb_node : int = 20
    apprentissage : list[sp] = [
        sp.csr_matrix(np.random.randint(600, size=(nb_node, nb_node))),  sp.csr_matrix(np.random.randint(600, size=(nb_node, nb_node)))
    ]
    apprentissage= [mat.astype(float) for mat in apprentissage]
    apprentissage = [mat - sp.diags(mat.diagonal()) for mat in apprentissage]

    test : list[sp] = [
        sp.csr_matrix(np.random.randint(600, size=(nb_node, nb_node))),  sp.csr_matrix(np.random.randint(600, size=(nb_node, nb_node)))
    ]
    test= [mat.astype(float) for mat in test]
    test = [mat - sp.diags(mat.diagonal()) for mat in test]


    G:int = len(apprentissage)
    list_K:list[int] = [2,3]
    
    K_star, log_tau, log_vect_pi, log_mat_lambda, list_critere_variationnel, list_icl, list_connectivite = optim_icl(apprentissage,nb_node,G,list_K)
    
    tau = np.exp(log_tau)
    partition = np.round(tau)    
    mat_lambda=np.exp(log_mat_lambda)

    rows = []
    
    for idx_matrice, mat in enumerate(test):
        mat = mat.tocoo()  # format COO = (row, col, data)
        
        for i, j, val in zip(mat.row, mat.col, mat.data):
            if  i!=j:
                rows.append({
                    "src": i,
                    "dest": j,
                    "count": val,
                    "idx_mat": idx_matrice
                })
    
    df_test = pd.DataFrame(rows)

### power computation graph
    
    list_vraisemblance_train_graphs = []
    for one_graph in apprentissage:
        log_v = ll_onegraph(log_vect_pi,partition,one_graph,log_mat_lambda,mat_lambda,K_star,nb_node)
        list_vraisemblance_train_graphs.append(log_v)
    
    pvalues_test=[]

    for one_graph in test:
        ll = ll_onegraph(log_vect_pi,partition,one_graph,log_mat_lambda,mat_lambda,K_star,nb_node)
        pvalue = 100*len([i for i in list_vraisemblance_train_graphs if i <= ll])/len(list_vraisemblance_train_graphs)
        pvalues_test.append(pvalue)

### power computation edge

    df_test = pvalues_edges(df_test, apprentissage, partition, log_vect_pi, log_mat_lambda, mat_lambda,nb_node, G)
    
    df_test.to_csv("pvalue_edge.csv", index=False)

### power computation degree

    df_degree_test = df_test.groupby(["idx_mat","src"]).size().reset_index(name="count")
    
    df_degree_test['count'] = df_degree_test['count'].fillna(0).astype(int)
    df_degree_test=pvalues_nodes(df_degree_test,apprentissage,log_tau,tau, mat_lambda,K_star,nb_node, G)
    df_degree_test.to_csv("pvalue_degree.csv", index=False)

    


    

