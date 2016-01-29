import numpy as np
from theano import tensor as tnsr
from theano import function




##-----model space-----  (uses tensordot)

#theano
rf_stack_tnsr = tnsr.tensor3('rf_stack_tnsr') ##G x stim_size x stim_size
feature_map_tnsr = tnsr.tensor4('feature_map_tnsr') ##T x D x stim_size x stim_size

apply_rf_to_feature_maps = function(inputs = [rf_stack_tnsr,feature_map_tnsr],
                                    outputs = tnsr.tensordot(rf_stack_tnsr,
							     feature_map_tnsr,
							     axes=[[1,2], [2,3]]))

#python use case
#model_space = apply_rf_to_feature_maps(rf_stack, feature_maps)

##-----prediction menu----- (uses batched_tensordot. not sure why this is necessary, but memory error if normal tensordot is used.)
#theano
model_space_tnsr = tnsr.tensor3('X')      ##model-space tensor: G x T x D
feature_weight_tnsr = tnsr.tensor3('NU')  ##feature weight tensor: G x D x V
prediction_menu_tnsr = tnsr.batched_tensordot(model_space_tnsr,
                                              feature_weight_tnsr,
                                              axes=[[2],[1]]) ##prediction tensor: G x T x V
bigmult = function([model_space_tnsr,feature_weight_tnsr], prediction_menu_tnsr)

##python use case
##prediction_menu = bigmult(model_space,feature_weights)  ##G x T x V


###-----error menu-----
##theano
voxel_data_tnsr = tnsr.matrix('voxel_data_tnsr')  ##voxel data tensor: T x V
diff = voxel_data_tnsr-prediction_menu_tnsr  ##difference tensor: (T x V) - (G x T x V) = (G x T x V)
sq_diff = (diff*diff).sum(axis=1) ##sum-sqaured-diffs tensor: G x V
sq_diff_func = function(inputs=[voxel_data_tnsr,prediction_menu_tnsr],
                        outputs = sq_diff)  

##python use case
##error_menu = sq_diff_func(voxel_data,prediction_menu)

###-----gradient menu-----
##theano
SQD_sum = sq_diff.sum()  ##<<this is critical
grad_SQD_wrt_NU = tnsr.grad(SQD_sum,feature_weight_tnsr) ##<<the summing trick above makes this easy. 
compute_grad = function(inputs = [voxel_data_tnsr,model_space_tnsr,feature_weight_tnsr],
                        outputs=grad_SQD_wrt_NU)


##--training function
def train_fwrf_model(model_space, voxel_data,initial_feature_weights,
                     early_stop_fraction = 0.2,
                     max_iters=100,
                     mini_batch_size = 0.1,
                     learning_rate=10**(-5),
                     voxel_binsize=100,
                     rf_grid_binsize = 200,
                     report_every = 10):

    ##basic dimenisions
    G,T,D = model_space.shape ##G = size of rf grid, T = number of trials (timepoints), D = number of feature weights
    _,V = voxel_data.shape ##V = number of voxels

    ##chunk up the voxels
    trnIdx = np.arange(0,T)
    early_stop_num = np.round(len(trnIdx)*early_stop_fraction).astype('int')
    voxel_bin_num = max(2,np.round(V/voxel_binsize))
    voxel_bins = np.linspace(0,V-1,num=voxel_bin_num,endpoint=True)

    ##chunk up the rf grids
    gdx = np.arange(0,G)
    rf_bin_num = np.round(G/rf_grid_binsize)
    rf_bins = np.linspace(0,G-1,num=rf_bin_num,endpoint=True)

    ##clock the whole function execution.
    big_start = time()
            
    ##prepare indices for data split
    perm_dx = np.random.permutation(trnIdx)
    validation_idx = perm_dx[0:early_stop_num]
    training_idx = perm_dx[early_stop_num:]
    validation_idx = np.atleast_2d(np.sort(validation_idx).astype('int')) ##1 x val_idx.shape[1]
    training_idx = np.atleast_2d(np.sort(training_idx).astype('int')) ##1 x trn_idx.shape[1]
    
    ##for storing the final model for each voxel
    final_rf = np.zeros(V).astype('int')
    final_feature_weights = np.zeros((D,V)).astype('float32')
    final_validation_loss = np.inf*np.ones(V)
    
    ####store error history of best rf model for each voxel
    best_error_history = np.zeros((max_iters,V))
    
    ##iterate over batches of voxels
    for v in range(len(voxel_bins)-1):
        
        ##indices for current batch of voxels
        v_idx = np.atleast_2d(np.arange(voxel_bins[v], voxel_bins[v+1]).astype('int')) ##1 x v_idx.shape[1]
        this_vox_batch_size = v_idx.shape[1]
        print '--------------voxels from %d to %d' %(v_idx[0,0],v_idx[0,-1])
        
        ##get data for these voxels
        this_trn_voxel_data = voxel_data[training_idx.T, v_idx]
        this_val_voxel_data = voxel_data[validation_idx.T, v_idx]
        print this_trn_voxel_data.shape
        print this_val_voxel_data.shape
        
        
        ##iterate over batches of rf models
        for g in range(len(rf_bins)-1):
            
            ##indices for current batch of rf models
            rf_idx = np.atleast_2d(np.arange(rf_bins[g], rf_bins[g+1]).astype('int')) ##1 x rf_idx.shape[1]
            this_rf_batch_size = rf_idx.shape[1]
            print '--------candiate rf models %d to %d' %(rf_idx[0,0],rf_idx[0,-1])
            
            ##slice model space for this batch of models / this batch of voxels
            this_trn_model_space = model_space[rf_idx.T,training_idx,:]
            this_val_model_space = model_space[rf_idx.T,validation_idx,:]
            print this_trn_model_space.shape
            print this_val_model_space.shape
#             1/0

            ##initialize best and current loss containers for this batch of voxels/models
            best_validation_loss = Inf*np.ones((this_rf_batch_size,this_vox_batch_size)) #rf_chunk x voxel_chunk
            this_validation_loss = np.zeros(best_validation_loss.shape)
            
            ##initialize best and current weight containers for this batch of voxels/models
            best_feature_weights = initial_feature_weights[rf_idx.flatten(),:,:]
            best_feature_weights = best_feature_weights[:,:,v_idx.flatten()]
            feature_weights = copy(best_feature_weights)
            
            
            ##initialize reports. so you can waste an entire afternoon watching your models train.
            iter_error = np.zeros((max_iters, this_vox_batch_size))
            bestie_change = np.zeros(max_iters)
            old_besties = np.zeros(this_vox_batch_size)
            
            ##initialize counters
            iters = 0
            start = time()
                        
            ##take gradient steps for a fixed number of iterations
            while (iters < max_iters):
                
                ##gradient: put a loop here over chunks of rf models to save on memory
                d_loss_wrt_params = compute_grad(this_trn_voxel_data,
                                                 this_trn_model_space,
                                                 feature_weights)
                
                
                ##update feature weights
                feature_weights -= learning_rate * d_loss_wrt_params
                
                ##predictions with updated feature weights
                prediction_menu = bigmult(this_val_model_space,
                                          feature_weights)

                ##updated loss
                this_validation_loss = sq_diff_func(this_val_voxel_data,
                                                    prediction_menu)
                
                ##if new loss minimum, save as best
                improved = this_validation_loss < best_validation_loss  ##rf batch x voxel batch
                imp = np.sum(improved)
                for ii in range(this_rf_batch_size):
                    best_validation_loss[ii,improved[ii,:]] = copy(this_validation_loss[ii,improved[ii,:]])
                    best_feature_weights[ii,:,improved[ii,:]] = copy(feature_weights[ii,:,improved[ii,:]])
                
                ##reporting business
                iter_error[iters,:]  = np.min(this_validation_loss,axis=0)
                besties = np.argmin(this_validation_loss,axis=0)
                bestie_change[iters] = np.sum(besties - old_besties)
                old_besties = copy(besties)
                if iters % report_every == 0:
                    print '-------'
                    print 'errors: %f' %(np.nanmean(iter_error[iters,:]))
                    print 'change in best rf: %f' %(bestie_change[iters])
                    print 'norm of feature weights: %f' %(np.sqrt(np.sum(feature_weights*feature_weights)))
                    print 'improvements: %d' %(imp)
                    print time()-start
                    start = time()
                
                ##update iteration
                iters += 1
            
            ##if the best of this batch of models has achieved new loss minimum, save it.
            for ii in range(this_vox_batch_size):
                best_of_batch_rf = np.argmin(best_validation_loss[:,ii]) ##index into current rf batch
                this_voxel = v_idx.flatten()[ii] ##total voxel index 
                if best_validation_loss[best_of_batch_rf,ii] < final_validation_loss[this_voxel]:
                    final_validation_loss[this_voxel] = copy(best_validation_loss[best_of_batch_rf,ii])
                    final_feature_weights[:,this_voxel] = copy(best_feature_weights[best_of_batch_rf,:,ii])
                    final_rf[this_voxel] = rf_idx[0,best_of_batch_rf]
                    best_error_history[:,this_voxel] = copy(iter_error[:,ii])

    print time()-big_start
    return final_validation_loss,final_feature_weights,final_rf,best_error_history
##----











