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
Pressure_Network = Pressure_overall_best

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

#error analysis: need to do this in non-normalized data
pressure_mean = scalingmatrix[4,1]
pressure_std = scalingmatrix[4,2]
#defining function to calculated error averages of datasets
function percenterr(data_set,model,outputmean,outputstd)
    test_percenterr_sum = 0.0
    for i in 1:length(data_set[:,1])
        #converting to pascals to avoid issues with 0 for percent error
        predicted = (model(data_set[i,[1,2,3]])[1] * outputstd) + outputmean
        actual = (data_set[i,4] * outputstd) + outputmean
        test_percenterr_sum += abs(predicted-actual)/actual
    end
    test_percenterr = 100*test_percenterr_sum/length(data_set[:,1])
    return test_percenterr
end

#using function
test_percenterr = percenterr(test_norm,Pressure_Network,pressure_mean,pressure_std)
train_percenterr = percenterr(train_norm,Pressure_Network,pressure_mean,pressure_std)
overall_err = percenterr(data_norm,Pressure_Network,pressure_mean,pressure_std)
print("Test Set %Err = $test_percenterr \n")
print("Train Set %Err = $train_percenterr \n")
print("Overall %Err = $overall_err \n")

#TODO:calculating percent change along constant mach/altitude lines
#grab all 0 and 0.99 mach values from training set
zero_fault = [];  ninetynine_fault = [];
for i in 1:length(train_data[:,1])
    if train_data[i,3] == 0
        push!(zero_fault, train_data[i,4])
    elseif train_data[i,3] == 0.99
        push!(ninetynine_fault, train_data[i,4])
    end
end
#calculating average percent change along const. mach/alt line
percent_change_avg = 0.0
for i in 1:length(zero_fault)
    global percent_change_avg += 100/length(zero_fault)*abs((zero_fault[i]
                            - ninetynine_fault[i])/ninetynine_fault[i])
end
print("Average percent change along constant mach/alt lines is $percent_change_avg%")


#Plotting Stuff
plotline = true;altitudeslice = true; #set these to true/false to adjust what is plotted
#lists to be plotted
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
machline = []; pressureline = []; faultline = [];
#specified_alt/mach for altitude slice and plotline respectively
alt_mean = scalingmatrix[1,1]; alt_std = scalingmatrix[1,2];
mach_mean = scalingmatrix[2,1]; mach_std = scalingmatrix[2,2];
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
            push!(model_SCAT,Pressure_Network(test_norm[i,1:3])[1])
        end
    else
        push!(mach_SCAT,mach)
        push!(fault_SCAT,fault)
        push!(pressure_SCAT,pressure)
        push!(model_SCAT,Pressure_Network(test_norm[i,1:3])[1])
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
