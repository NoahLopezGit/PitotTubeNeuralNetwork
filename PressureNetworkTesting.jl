#testing for pressure network
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Plots
using BSON: @load
using Random
using Statistics

#load network to test
@load "Models/Pressure_overall_best.bson" Pressure_overall_best
Pressure_Model = Pressure_overall_best
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

alt_mean = scalingmatrix[1,1]; alt_std = scalingmatrix[1,2];
mach_mean = scalingmatrix[2,1]; mach_std = scalingmatrix[2,2];
fault_mean = scalingmatrix[3,1];fault_std = scalingmatrix[3,2]
pressure_mean = scalingmatrix[4,1]; pressure_std = scalingmatrix[4,2]
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
        MSE_sum += 1/N * (data_set[i,4] - model(data_set[i,1:3])[1])^2.0
    end
    return sqrt(MSE_sum)  # no norm needed bc data is already normalized
end



print("Error Anlysis")
print("\nOverall root mean squared error: ")
print(RMSE(data_norm,Pressure_Model))
print("\nTest root mean squared error: ")
print(RMSE(test_norm,Pressure_Model))
print("\nTrain root mean squared error: ")
print(RMSE(train_norm,Pressure_Model))

#grab all 0 and 0.99 mach values from training set
#TODO: this is like interval of prediction (goes +/- 0.23ish)
#"interval of prediction" for this model is like 10x smaller than fault model
zero_fault = [];  ninetynine_fault = [];

for i in 1:length(train_norm[:,1])
    if train_data[i,3] == 0.0
        push!(zero_fault, train_norm[i,4])
    elseif train_data[i,3] == 0.99
        push!(ninetynine_fault, train_norm[i,4])
    end
end
N = length(zero_fault)
#calculating average percent change along const. mach/alt line
MSE_change = 0.0
for i in 1:N
    global MSE_change += 1/N * (ninetynine_fault[i] - zero_fault[i])^2.0
end
RMSE_change = sqrt(MSE_change)
print("\nAverage RMSE change along constant mach/alt lines is: $RMSE_change")


#Plotting Stuff
plotline = true;altitudeslice = true; #set these to true/false to adjust what is plotted
#lists to be plotted
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
machline = []; pressureline = []; faultline = [];
#specified_alt/mach for altitude slice and plotline respectively

specified_alt_tmp = 1525
specified_alt = (specified_alt_tmp - alt_mean)/alt_std
specified_mach = (0.49 - mach_mean)/mach_std
#organizing plotting data
for i in 1:length(test_norm[:,1])
    mach = test_norm[i,2]
    fault = test_norm[i,3]
    pressure = test_norm[i,4]
    alt = test_norm[i,1]
    if altitudeslice == true
        if alt == specified_alt
            push!(mach_SCAT,mach)
            push!(fault_SCAT,fault)
            push!(pressure_SCAT,pressure)
            push!(model_SCAT,Pressure_Model(test_norm[i,1:3])[1])
        end
    else
        push!(mach_SCAT,mach)
        push!(fault_SCAT,fault)
        push!(pressure_SCAT,pressure)
        push!(model_SCAT,Pressure_Model(test_norm[i,1:3])[1])
    end
    #plot line stuff
    if mach == specified_mach && alt == specified_alt
        push!(machline, mach)
        push!(pressureline, pressure)
        push!(faultline, fault)
    end
end
if altitudeslice == true
    title = "$specified_alt_tmp"
else
    title = "All"
end
#creating 3d scatter for showing network results
scatter(fault_SCAT,mach_SCAT,pressure_SCAT,label="Actual",
        title="Predictions with Testing Data [$title]",
        xlabel="Fault Parameter",
        ylabel="Mach",
        zlabel="Pressure")
scatter!(fault_SCAT,mach_SCAT,model_SCAT,label="Predicted")
#plotline
if plotline == true
    plot!(faultline,machline,pressureline, label="Constant Mach/Alt")
end
current()
