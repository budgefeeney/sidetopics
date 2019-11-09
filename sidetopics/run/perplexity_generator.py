'''
Reads in all outputs from model cross-validation runs and (re-)generates
train and query perplexity scores.

Created on 4 Mar 2014

@author: bryanfeeney
'''



def generate_perplexities_stm(W, X, topicCounts, latentDims, outputFilesDir):
    P = len(latentDims)
    K = len(topicCounts)
    
    perps  = { \
        'train' : { \
            'bouchard' : { \
                1 : np.zeros((P,K)), \
                3 : np.zeros((P,K)), \
                5 : np.zeros((P,K))
            }, \
            'bohning' : { \
                1 : np.zeros((P,K)), \
                3 : np.zeros((P,K)), \
                5 : np.zeros((P,K)) \
            } \
        }, \
        'query' : { \
            'bouchard' : { \
                1 : np.zeros((P,K)), \
                3 : np.zeros((P,K)), \
                5 : np.zeros((P,K))
            }, \
            'bohning' : { \
                1 : np.zeros((P,K)), \
                3 : np.zeros((P,K)), \
                5 : np.zeros((P,K)) \
            } \
        }, \
    }
          
    for bound in bounds:
        log_likely = stm_yv_bou_log_likelihood \
            if bound == 'bouchard' \
            else stm_yv_boh_log_likelihood
        
        for k_id in range(K):
            k = topicCounts[k_id]
            for p_id in range(P):
                p = latentDims[p_id]
                orders, _, _, model, trainTopics, queryTopics = \
                    load_folds (outputFilesDir, StmYvAlgorName, bound, k, p)
            
                if len(orders) == 0:
                    continue
            
                W_ro, X_ro = reorder_dataset (W, X, orders[0]) # same across all folds  
                tlikelies     = []
                tword_counts  = []
                
                qlikelies     = []
                qword_counts  = []
            
                foldSize  = ceil(D / ExpectedFoldCount)
                querySize = foldSize
                trainSize = D - querySize
            
                for fold in range(len(model)):
                    start = fold * foldSize
                    end   = start + trainSize
                    
                    trainSet = np.arange(start,end) % D
                    querySet = np.arange(end, end + querySize) % D
        
                    W_train, W_query = W_ro[trainSet,:], W_ro[querySet,:]
                    # X_train, X_query = X_ro[trainSet,:], X_ro[querySet,:]
                    
                    #sys.stdout.write("Train Likely: Bound = %s K = %d, P = %d, Fold = %d\n" % (bound, k, p, fold))
                    #sys.stdout.write("\tW_train.D = %d, W_train.sum() = %.0f, trainTopics.means.D = %d\n" % (W_train.shape[0], W_train.data.sum(), trainTopics[fold].means.shape[0]))
                    #sys.stdout.write("\tLog Likely = %f\n" % (log_likely(W_train, model[fold], trainTopics[fold])))
                    #sys.stdout.flush()
                    tlikelies.append(log_likely(W_train, model[fold], trainTopics[fold]))
                    tword_counts.append (W_train.data.sum())
                    
                    #sys.stdout.write("Query Likely: Bound = %s K = %d, P = %d, Fold = %d\n" % (bound, k, p, fold))
                    #sys.stdout.write("\tW_qury.D = %d, W_query.sum() = %.0f, trainTopics.means.D = %d\n" % (W_query.shape[0], W_query.data.sum(), queryTopics[fold].means.shape[0]))
                    #sys.stdout.write("\tLog Likely = %f\n" % (log_likely(W_query, model[fold], queryTopics[fold])))
                    #sys.stdout.flush()
                    qlikelies.append(log_likely(W_query, model[fold], queryTopics[fold]))
                    qword_counts.append (W_query.data.sum())
            
                if len(qlikelies) >= 0:
                    perps['query'][bound][1][p_id, k_id] = perplexity (qlikelies[0], qword_counts[0])
                if len(qlikelies) >= 3:
                    perps['query'][bound][3][p_id, k_id] = perplexity (sum(qlikelies[:3]), sum(qword_counts[:3]))
                if len(qlikelies) >= 5:
                    perps['query'][bound][5][p_id, k_id] = perplexity (sum(qlikelies), sum(qword_counts))
                if len(tlikelies) >= 0:
                    perps['train'][bound][1][p_id, k_id] = perplexity (tlikelies[0], tword_counts[0])
                if len(tlikelies) >= 3:
                    perps['train'][bound][3][p_id, k_id] = perplexity (sum(tlikelies[:3]), sum(tword_counts[:3]))
                if len(tlikelies) >= 5:
                    perps['train'][bound][5][p_id, k_id] = perplexity (sum(tlikelies), sum(tword_counts))
            
            

if __name__ == '__main__':
    pass
    pass
    pass
    pass
    pass
    pass