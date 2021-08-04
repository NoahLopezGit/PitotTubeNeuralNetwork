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
@load "Models/Fault_Model.bson" Fault_Model

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


#Normalized Root Mean Squared Error
function RMSE(data_set,model)
    #calculate root mean squared error
    N = length(data_set[:,1])
    print(N,'\n')
    MSE_sum = 0.0
    for i in 1:N
        MSE_sum += 1/N * (data_set[i,3] - model(data_set[i,[1,2,4]])[1])^2.0
    end
    return sqrt(MSE_sum)  # no norm needed bc data is already normalized
end

print("Error Anlysis")
print("\nOverall root mean squared error: ")
print(RMSE(data_norm,Fault_Model))
print("\nTest root mean squared error: ")
print(RMSE(test_norm,Fault_Model))
print("\nTrain root mean squared error: ")
print(RMSE(train_norm,Fault_Model))
#TODO: represent interval of prediction (normalied goes -2 to 0.5 ish)

#plotting
alt_mean = scalingmatrix[1,1]
alt_std = scalingmatrix[1,2]
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
specified_alt = (12000 - alt_mean)/alt_std
for i in 1:length(test_norm[:,1])
    if test_norm[i,1] == specified_alt
        push!(mach_SCAT,test_norm[i,2])
        push!(fault_SCAT,test_norm[i,3])
        push!(pressure_SCAT,test_norm[i,4])
        push!(model_SCAT,Fault_Model(test_norm[i,[1,2,4]])[1])
    end
end
#creating 3d scatter for showing network results
scatter(pressure_SCAT,mach_SCAT,fault_SCAT,label="Actual",
        title="Predictions with Testing Data [All]",
        xlabel="Pressure",
        ylabel="Mach",
        zlabel="Fault Parameter")
scatter!(pressure_SCAT,mach_SCAT,model_SCAT,label="Predicted")
