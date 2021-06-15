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
    close(f1)
    data = []
    regvec = zeros(length(tempdata),
                   length(split(tempdata[1], "\t")))
    #parsing datavector to float array... not regularized
    for i in 1:length(tempdata)
        regvec[i,:] =  parse.(Float64,split(tempdata[i], "\t"))
    end
    return regvec
end

#normalizing data
function norm_data(data, scalingmatrix=nothing)
    #getting regularization values and normalizing data
    regvec = data
    normvec = zeros(length(regvec[:,1]),
                    length(regvec[1,:]))
    #checkign if optional scalingmatrix was passed, if not will calc mean and std
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

#getting training data
fname = "Dataset/training.txt"
train_data = get_data(fname)
train_norm, scalingmatrix = norm_data(train_data)
#converting data to format which works with network
data = []
for i in 1:length(train_norm[:,1])
    push!(data,(train_norm[i,1:3],train_norm[i,4]))
end
#network initialization function (allows for easier iterations)
function networkitr(data,Q,wd,iterations)
    #model... must adjust if you want a different structure
    itrmodel = Chain(  Dense(3,Q),  #σ(w1*x1 + w2*x2 + ...)
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
for q in [10,20,30,40]
    for wd in [0,0.1,0.001,0.0001,0.00001,0.000001,0.0000001]
        #training netowrk iteration
        print("\nTesting q=$q,wd=$wd\n")
        global iteration_best, lowestmse_itr = networkitr(data,q,wd,1000)
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
#getting test dataset
fname = "Dataset/testing.txt"
test_data = get_data(fname)
test_norm,scalingmatrix = norm_data(test_data,scalingmatrix)
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
#specified_alt = 1525/12000 #TODO:update for normalization
for i in 1:length(test_norm[:,1])
    #if datapoint[1] == specified_alt
        push!(mach_SCAT,test_norm[i,2])
        push!(fault_SCAT,test_norm[i,3])
        push!(pressure_SCAT,test_norm[i,4])
        push!(model_SCAT,overall_best(test_norm[i,1:3])[1])
    #end
end
#creating 3d scatter for showing network results
scatter(fault_SCAT,mach_SCAT,pressure_SCAT,label="Actual",
        title="Predictions with Testing Data [1525m]",
        xlabel="Fault Parameter",
        ylabel="Mach",
        zlabel="Pressure (normalized)")
scatter!(fault_SCAT,mach_SCAT,model_SCAT,label="Predicted")
