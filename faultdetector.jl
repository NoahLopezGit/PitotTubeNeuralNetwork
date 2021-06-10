cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux
using Plots
using Flux: @epochs, @show
using BSON: @save
using Random

fname = "training.txt"
f1 = open(fname)
tempdata = readlines(f1)
close(f1)
data = []
Pmax = 0.0
for i in 1:length(tempdata)
    global Pmax = max(parse(Float64,split(tempdata[i], "\t")[4]), Pmax)
end
for i in 1:length(tempdata)
    alt = parse(Float64,split(tempdata[i], "\t")[1])/12000  #normalize by alt max
    mach = parse(Float64,split(tempdata[i], "\t")[2])
    fault = parse(Float64,split(tempdata[i], "\t")[3])
    pressure = parse(Float64,split(tempdata[i], "\t")[4])/Pmax
    push!(data,([alt, mach, pressure], fault))
end

Q = 48
model = Chain(  Dense(3,2*Q),  #σ(w1*x1 + w2*x2 + ...)
                Dense(2*Q,Q,celu),
                Dense(Q,Q,celu),
                Dense(Q,1,σ)) # 1 output connect to dim = Q HL

loss(x, y) = Flux.msle(model(x), y) # cost function
opt = ADAM()
para = Flux.params(model) # variable to represent all of our weights and biases
#=
for 500 iterations of our entire data set (@epochs 500) model is "trained"
Flux.train!( cost function, weights and biases, training data, optimizer setting)
=#


err = []
a = 0
mseplot= []
for i in 1:length(data)
    global a = 0
    global a = a + loss(data[i][1],data[i][2])
end
a = a/length(data)
data = Random.shuffle(data)
while a > 5e-5
    global a = 0
    global data = Random.shuffle(data)
    local trainset = data[1:32]
    Flux.train!(loss, para, trainset, opt)
    for i in 1:length(data)
        global a =  a + loss(data[i][1],data[i][2])
    end
    global a = a/length(data)
    print("Error $a \n")
    #push!(mseplot,a)
end

fname = "testing.txt"
f2 = open(fname)
tempdata = readlines(f2)
close(f2)
testdata = []
Pmax = 0.0
for i in 1:length(tempdata)
    global Pmax = max(parse(Float64,split(tempdata[i], "\t")[4]), Pmax)
end
for i in 1:length(tempdata)
    alt = parse(Float64,split(tempdata[i], "\t")[1])/12000  #normalize by alt max
    mach = parse(Float64,split(tempdata[i], "\t")[2])
    fault = parse(Float64,split(tempdata[i], "\t")[3])
    pressure = parse(Float64,split(tempdata[i], "\t")[4])/Pmax
    push!(testdata, (alt,mach,fault,pressure))
end

mach_SCAT = []
pressure_SCAT = []
fault_SCAT = []
model_SCAT = []
specified_alt = 6180.56/12000
for datapoint in testdata
    if datapoint[1] == specified_alt
        push!(mach_SCAT,datapoint[2])
        push!(pressure_SCAT,datapoint[4])
        push!(fault_SCAT,datapoint[3])
        push!(model_SCAT,model([datapoint[1],datapoint[2],datapoint[4]])[1])
    end
end
scatter(pressure_SCAT,mach_SCAT,fault_SCAT,label="Actual",
        title="Predicted Fault vs Actual Fault [6180.56 m]",
        xlabel="Pressure (Normalized)",
        ylabel="Mach Number",
        zlabel="Fault Parameter")

scatter!(pressure_SCAT,mach_SCAT,model_SCAT,label="Predicted")
