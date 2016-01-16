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

#python
model_space = apply_rf_to_feature_maps(rf_stack, feature_maps)

##-----prediction menu----- (uses batched_tensordot. not sure why this is necessary, but memory error if normal tensordot is used.)
#theano
model_space_tnsr = tnsr.tensor3('X')    ##model-space tensor: G x T x D
feature_weight_tnsr = tnsr.tensor3('NU')  ##feature weight tensor: G x D x V
prediction_menu_tnsr = tnsr.batched_tensordot(model_space_tnsr,
                                              feature_weight_tnsr,
                                              axes=[[2],[1]]) ##prediction tensor: G x T x V
bigmult = function([model_space_tnsr,feature_weight_tnsr], prediction_menu_tnsr)

#python
prediction_menu = bigmult(model_space,feature_weights)  ##G x T x V


##-----error menu-----
#theano
voxel_data_tnsr = tnsr.matrix('voxel_data_tnsr')  ##voxel data tensor: T x V

diff = voxel_data_tnsr-prediction_menu_tnsr  ##difference tensor: (T x V) - (G x T x V) = (G x T x V)
sq_diff = (diff*diff).sum(axis=1) ##sum-sqaured-diffs tensor: G x V
sq_diff_func = function(inputs=[voxel_data_tnsr,prediction_menu_tnsr],
                        outputs = sq_diff)  

#python
error_menu = sq_diff_func(voxel_data,prediction_menu)

##-----gradient menu-----
#theano
SQD_sum = sq_diff.sum()  ##<<this is critical
grad_SQD_wrt_NU = tnsr.grad(SQD_sum,feature_weight_tnsr) ##<<the summing trick above makes this easy. 
compute_grad = function(inputs = [voxel_data_tnsr,model_space_tnsr,feature_weight_tnsr],
                        outputs=grad_SQD_wrt_NU)



##-----training loop----
#python
best_params = None
params = np.zeros(feature_weights.shape).astype('float32')
iters = 0
this_validation_loss = Inf
best_validation_loss = Inf
iter_error = np.zeros((max_iters,V))
start = time()
while (iters < max_iters):
    best_validation_loss = this_validation_loss
    
       
    ##gradient
    d_loss_wrt_params = compute_grad(voxel_data[training_idx, :],
                                model_space[:,training_idx,:],
                                params)  
    ##update
    params -= learning_rate * d_loss_wrt_params

    ##predictions with update params
    prediction_menu = bigmult(model_space[:,validation_idx,:],
                         params)

    ##update loss
    this_validation_loss = sq_diff_func(voxel_data[validation_idx, :],
                                        prediction_menu)

    
    iter_error[iters,:]  = np.min(this_validation_loss,axis=0)
    besties = np.argmin(this_validation_loss,axis=0)
    if iters % 250 == 0:
        print '-------'
        #print 'errors: %s' %(iter_error[iters,:])
        #print 'besties: %s' %(besties,)
#         corr_list = []
#         for i,b in enumerate(besties):
#             corr_list.append(np.corrcoef(feature_weights[b,:,i],params[b,:,i])[0,1])
#         print 'corrs: %s' %(corr_list,)
        print '--------------------------------------------------'
        
        
    
    iters += 1
        
print time()-start