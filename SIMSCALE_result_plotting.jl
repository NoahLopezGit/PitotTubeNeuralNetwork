cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Plots


##This code plots the simscale results
function get_data(filename)
    f1 = open(filename)
    tempdata = readlines(f1)
    print(tempdata[1],'\n')
    #use this for "Dataset/FINAL_DATA_.txt" for other files may not
    #tempdata[1] = tempdata[1][4:end]  #cutting of beggining bc some weird chars
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


#getting simscale results
fname = "Dataset/FINAL_DATA_reorganized.txt"
data_tmp = get_data(fname)

#series color is gradient based on altitude
alt=zeros(1,length(data_tmp[:,1]));mach=zeros(1,length(data_tmp[:,1]));
fault=zeros(1,length(data_tmp[:,1]));pressure=zeros(1,length(data_tmp[:,1]));

machline=[];faultline=[];pressureline=[];
specified_mach = 0.3;specified_alt=1525.0
for i in 1:length(data_tmp[:,1])
    alt[1,i] = data_tmp[i,1]
    mach[1,i] = data_tmp[i,2]
    fault[1,i] = data_tmp[i,3]
    pressure[1,i] = data_tmp[i,4]/1000.0
    if mach[1,i] == specified_mach && alt[1,i] == specified_alt
        push!(machline, mach[1,i])
        push!(pressureline, pressure[1,i])
        push!(faultline, fault[1,i])
    end
end

plotly()
scatter(fault,mach,pressure,
        title="SIMSCALE Results",
        xlabel="Fault",
        ylabel="Mach",
        zlabel="Pressure (kPa)",
        c=:alpine,
        colorbar_title="Alt",
        marker_z=alt,
        clims=(1525.0,12000.0),
        markersize=2,
        label=nothing)
plot!(faultline,machline,pressureline,
        label="Constant Mach/Alt",
        linecolor=:red)

#second plot
fault=[];mach=[];alt=[];pressure=[];

machline=[];faultline=[];pressureline=[];
specified_mach = 0.3;specified_alt=1525.0
for i in 1:length(data_tmp[:,1])
    if data_tmp[i,1]==specified_alt
        push!(alt, data_tmp[i,1])
        push!(mach, data_tmp[i,2])
        push!(fault, data_tmp[i,3])
        push!(pressure, data_tmp[i,4]/1000.0)
    end
    if data_tmp[i,2] == specified_mach && data_tmp[i,1] == specified_alt
        push!(machline, data_tmp[i,2])
        push!(pressureline, data_tmp[i,4]/1000.0)
        push!(faultline, data_tmp[i,3])
    end
end
gr()
scatter(fault,mach,pressure,
    title="Altitude Slice (1525m)",
    xlabel="Fault",
    ylabel="Mach",
    zlabel="Pressure (kPa)")
plot!(faultline,machline,pressureline,
        label="Constant Mach/Alt",
        linecolor=:red)
