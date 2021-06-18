cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Plots
using Flux: @epochs, @show
using BSON: @save
using Random
using Statistics


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
#converting data to format which works with network
data = []
for i in 1:length(train_norm[:,1])
    #adjust x and y for fault detection here
    push!(data,(train_norm[i,[1,2,3]],train_norm[i,4]))
end


#network initialization function (allows for easier iterations)
function networkitr(data,Q,wd,iterations)
    #model... must adjust if you want a different structure
    itrmodel = Chain(  Dense(3,Q),  #σ(w1*x1 + w2*x2 + ...)
                Dense(Q,Q,gelu),
                Dense(Q,Q,gelu),
                Dense(Q,Q,gelu),
                Dense(Q,Q,gelu),
                Dense(Q,Q,gelu),
                Dense(Q,Q,gelu),
                Dense(Q,1)) # 1 output connect to dim = Q H
    opt = ADAM()
    para = Flux.params(itrmodel) # variable to represent all of our weights and biases
    l1(x) = sum(x .^ 2)
    loss(x, y) = Flux.mse(itrmodel(x), y) + wd * sum(l1, para) # cost function
    loss_wout_reg(x,y) = Flux.mse(itrmodel(x), y)
    #=
    for 500 iterations of our entire data set (@epochs 500) model is "trained"
    Flux.train!( cost function, weights and biases, training data, optimizer setting)
    =#
    lowestmse = 1.0
    for i in 1:iterations
        trainset = Random.shuffle(data)[1:64]
        Flux.train!(loss, para, trainset, opt)
        Err = 0.0
        for datapoint in data
            Err += loss_wout_reg(datapoint[1],datapoint[2])
        end
        Err = Err/length(data)
        #calculating lowestmse
        if lowestmse > Err
            global iterationbest = itrmodel #setting best model to lowestmse
            lowestmse = Err
        end
        #print iteration data
        print("\rError $lowestmse Itr $i/$iterations         ")
    end
    return iterationbest, lowestmse
end
#iterating training diff networks
lowestmse_overall = 1.0
for q in [8,16,32]
    for wd in [0,0.00001,0.000001,0.0000001]
        #training netowrk iteration
        print("\nTesting q=$q,wd=$wd\n")
        global iteration_best, lowestmse_itr = networkitr(data,q,wd,2000)
        if lowestmse_itr < lowestmse_overall
            print("\nlowestmse = $lowestmse_itr  < best overall = $lowestmse_overall\n")
            print("Replacing Network with best network: ")
            global best_string = "$q Nodes, $wd regularization parameter\n"
            print(best_string)
            global overall_best = iteration_best
            global lowestmse_overall = lowestmse_itr
        end
    end
end
print("best networkw was: ", best_string, "with Error $lowestmse_overall")
#getting test dataset
plotline = false;altitudeslice = false; #set these to true/false to adjust what is plotted
#lists to be plotted
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
machline = []; pressureline = []; faultline = [];
#getting data to be plotted
test_norm,scalingmatrix = norm_data(test_data,scalingmatrix)
#specified_alt/mach for altitude slice and plotline respectively
alt_mean = scalingmatrix[1,1]; alt_std = scalingmatrix[1,2];
mach_mean = scalingmatrix[2,1]; mach_std = scalingmatrix[2,2];
specified_alt = (1525 - alt_mean)/alt_std
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
            push!(model_SCAT,overall_best(test_norm[i,1:3])[1])
        end
    else
        push!(mach_SCAT,mach)
        push!(fault_SCAT,fault)
        push!(pressure_SCAT,pressure)
        push!(model_SCAT,overall_best(test_norm[i,1:3])[1])
    end
    #plot line stuff
    if mach == specified_mach && alt == specified_alt
        push!(machline, mach)
        push!(pressureline, pressure)
        push!(faultline, fault)
    end
end
#creating 3d scatter for showing network results
scatter(fault_SCAT,mach_SCAT,pressure_SCAT,label="Actual",
        title="Predictions with Testing Data [All]",
        xlabel="Fault Parameter",
        ylabel="Mach",
        zlabel="Pressure")
scatter!(fault_SCAT,mach_SCAT,model_SCAT,label="Predicted")
#plotline
if plotline == true
    plot!(faultline,machline,pressureline, label="Constant Mach/Alt")
end
current()

#error analysis: need to do this in non-normalized data
pressure_mean = scalingmatrix[4,1]
pressure_std = scalingmatrix[4,2]

test_percenterr_sum = 0.0
for i in 1:length(test_norm[:,1])
    #converting to pascals to avoid issues with 0 for percent error
    predicted = (overall_best(test_norm[i,[1,2,3]])[1] * pressure_std) + pressure_mean
    actual = (test_norm[i,4] * pressure_std) + pressure_mean
    global test_percenterr_sum += abs(predicted-actual)/actual
end
test_percenterr = 100*test_percenterr_sum/length(test_norm[:,1])

train_percenterr_sum = 0.0
for i in 1:length(train_norm[:,1])
    #converting to pascals to avoid issues with 0 for percent error
    predicted = (overall_best(train_norm[i,[1,2,3]])[1] * pressure_std) + pressure_mean
    actual = (train_norm[i,4] * pressure_std) + pressure_mean
    global train_percenterr_sum += abs(predicted-actual)/actual
end
train_percenterr = 100*train_percenterr_sum/length(train_norm[:,1])

print("Test Set %Err = $test_percenterr \n")
print("Train Set %Err = $train_percenterr \n")

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
