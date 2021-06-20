#error analysis of fault prediction network
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Plots
using BSON: @load
using Random
using Statistics

#load network to test
@load "Models/Fault4_Fault_Model.bson" Fault_Model

#function for parsing data
function get_data(filename)
    f1 = open(filename)
    tempdata = readlines(f1)
    tempdata[1] = tempdata[1][4:end]  #cutting of beggining bc some weird chars
    close(f1)
    data = []
    regvec = zeros(length(tempdata),
                   length(split(tempdata[1], ",")))
    #parsing datavector to float array... not regularized
    for i in 1:length(tempdata)
        regvec[i,:] =  parse.(Float64,split(tempdata[i], ","))
    end
    return regvec
end

#normalizing data
function norm_data(data, scalingmatrix=nothing)
    #getting regularization values and normalizing data
    regvec = data
    normvec = zeros(length(regvec[:,1]),
                    length(regvec[1,:]))
    #checking if optional scalingmatrix was passed, if not will calc mean and std
    if scalingmatrix == nothing
        scalingmatrix = zeros(length(regvec[1,:]),2)
        for i in 1:length(regvec[1,:])
            scalingmatrix[i,1] = Statistics.mean(regvec[:,i]) #first index is mean
            scalingmatrix[i,2] = Statistics.std(regvec[:,i])  #second index is std
        end
        for i in 1:length(regvec[1,:])
            #normalizing regvec
            normvec[:,i] = (regvec[:,i] .- scalingmatrix[i,1]) ./ scalingmatrix[i,2]
        end
    else
        #for case of normalizing test data by training std and mean
        for i in 1:length(regvec[1,:])
            #normalizing regvec
            normvec[:,i] = (regvec[:,i] .- scalingmatrix[i,1]) ./ scalingmatrix[i,2]
        end
    end
    return normvec, scalingmatrix
end

#function to convert vector of vector to matrix
function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end

#getting training data
fname = "Dataset/FINAL_DATA.txt"
data_tmp = get_data(fname)
#seperating test and train data
train_data_tmp = []; test_data_tmp = [];
for i in 1:length(data_tmp[:,1])
    if data_tmp[i,2] in [0.3,0.38,0.46,0.53,0.61,0.65] #training case
        push!(train_data_tmp, data_tmp[i,:])
    else #testing case
        push!(test_data_tmp, data_tmp[i,:])
    end
end
train_data = vecvec_to_matrix(train_data_tmp)
test_data = vecvec_to_matrix(test_data_tmp)
#normalizing data
train_norm, scalingmatrix = norm_data(train_data)
test_norm, scalingmatrix = norm_data(test_data,scalingmatrix)
data_norm, scalingmatrix = norm_data(data_tmp,scalingmatrix)
#converting data to format which works with network
data = []
for i in 1:length(train_norm[:,1])
    #adjust x and y for fault detection here
    push!(data,(train_norm[i,[1,2,3]],train_norm[i,4]))
end


#TODO: %err is tricky around 0 find a better way to present the error

#function to calculate mean error and std to give 95% confidence interval
function erroranal(dataset,model,faultstd,faultmean)
    errorlist = []
    for i in 1:length(dataset[:,1])
        #adjust x and y for fault detection here
        prediction = (model(dataset[i,[1,2,4]])[1] * faultstd) + faultmean
        actual = (dataset[i,3] * faultstd) + faultmean
        push!(errorlist, prediction - actual)
    end
    return errorlist
end

faultstd = scalingmatrix[3,2];faultmean = scalingmatrix[3,1];
totalerror = erroranal(data_norm,Fault_Model,faultstd,faultmean)
errormean = Statistics.mean(abs.(totalerror))
errorstd = Statistics.std(abs.(totalerror))
ninetyfive_confidence = errormean + 2 * errorstd
print("This network has an average error of $errormean.\n",
        "with a standard deviation of $errorstd.\n",
        "This means the network can predict the fault parameter to:\n",
        "+/-$ninetyfive_confidence with a confidence of 95%")
