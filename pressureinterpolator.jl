cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Plots
using Flux: @epochs, @show
using BSON: @save
using Random
using Statistics

fname = "Dataset/training.txt"

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

train_data = get_data(fname)
train_norm, scalingmatrix = norm_data(train_data)
#converting data to format which works with network
data = []
for i in 1:length(train_norm[:,1])
    push!(data,(train_norm[i,1:3],train_norm[i,4]))
end

#initializing network
Q = 50
model = Chain(  Dense(3,Q),  #σ(w1*x1 + w2*x2 + ...)
                Dense(Q,Q,celu),
                Dense(Q,Q,σ),
                Dense(Q,1,σ)) # 1 output connect to dim = Q HL

loss(x, y) = Flux.mse(model(x), y) # cost function
opt = ADAM()
para = Flux.params(model) # variable to represent all of our weights and biases
#=
for 500 iterations of our entire data set (@epochs 500) model is "trained"
Flux.train!( cost function, weights and biases, training data, optimizer setting)
=#


err = []
a = 0
mseplot = []
for i in 1:length(data)
    global a = 0
    global a = a + loss(data[i][1],data[i][2])
end
a = a/length(data)
for j in 1:1000
    global a = 0
    global data = Random.shuffle(data)
    Flux.train!(loss, para, data, opt)
    for i in 1:length(data)
        global a =  a + loss(data[i][1],data[i][2])
    end
    global a = a/length(data)
    print("Error $a \n")
    push!(mseplot,a)
end

#getting test dataset
fname = "Dataset/testing.txt"
test_data = get_data(fname)
test_norm,scalingmatrix = norm_data(test_data,scalingmatrix)
mach_SCAT = [];pressure_SCAT = [];fault_SCAT = [];model_SCAT = [];
specified_alt = 1525/12000 #TODO:update for normalization
for i in 1:length(test_norm[:,1])
    #if datapoint[1] == specified_alt
        push!(mach_SCAT,test_norm[i,2])
        push!(fault_SCAT,test_norm[i,3])
        push!(pressure_SCAT,test_norm[i,4])
        push!(model_SCAT,model(test_norm[i,1:3])[1])
    #end
end
#creating 3d scatter for showing network results
scatter(fault_SCAT,mach_SCAT,pressure_SCAT,label="Actual",
        title="Predictions with Testing Data [1525m]",
        xlabel="Fault Parameter",
        ylabel="Mach",
        zlabel="Pressure (normalized)")
scatter!(fault_SCAT,mach_SCAT,model_SCAT,label="Predicted")
#plotting MSE over training iterations
plot(mseplot,title="Mean Squared Error vs Epochs",xlabel="MSE",ylabel="EPOCHS",label="MSE")
