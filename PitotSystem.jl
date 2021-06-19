cd(@__DIR__)
using Pkg
Pkg.activate(".")

using F16Model, LinearAlgebra, Plots
using BSON: @load
using Flux
@load "Models/Pressure_overall_best.bson" Pressure_overall_best
pressuremodel = Pressure_overall_best
@load "Models/Fault4_Fault_Model.bson" Fault_Model
faultmodel = Fault_Model


#just copied and pasted these... might want to automate in future
fault_std = 0.310076;fault_mean = 0.741667;
alt_std = 3345.34; alt_mean = 6762.5;
mach_std = 0.123024;mach_mean = 0.488333;
pressure_std = 24622.9; pressure_mean = 53718.4;

#=
Aircraft states are:
npos = North position in ft
epos = East position in ft
alt = height/altitude in ft
phi = Euler angle in rad TODO is this equivalent to Yaw,Pitch,Roll?
theta = Euler angle in rad
psi = Euler angle in rad
Vt = Flight velocity in ft/s
alp = Angle of attack in rad
bet = Side slip angle in rad
p = roll rate
q = pitch rate
r = yaw rate

Aircraft controls are:
T = Thrust in lbs
dele = Elevator angle in deg
dail = Aileron angle in deg
drud = Rudder angle in deg
dlef = Leading edge flap angle in deg
=#

# ===================================================
# Trim the aircraft for steady-level flight at h0,V0
# xbar = trim state
# ubar = trim control
# status = status of the optimization, status = 0 means optimization found solution and (xbar, ubar) defines a valid trim  point/
# prob = data structure from IpOpt.

h0 = 15000; # ft
Vt0 = 500;   # ft/s
xbar, ubar, status, prob = F16Model.Trim(h0,Vt0); # Default is steady-level

# ===================================================
# Linearize about some trim point (x0,u0)
Ac, Bc = F16Model.Linearize(xbar,ubar); # Get continuous-time model.
nx,nu = size(Bc);

# to indclude gps we want Npos and Epos (x[1,2])
ix = [1,2,3,5,7,8,11]; # States I want...  now including Npos (1) and Epos (2)
iu =[2]; # Control variables I want, this how the disturbance comes in.

# Pull out longitudinal model
Al = Ac[ix,ix];
Bl = Bc[ix,iu];

# ============= TO DO #1 =========
# 1. Define the sensor model and write continuous time dynamics w.r.t z (the augmented state var)
# 2. Discretize the model using code below.
Vbar = xbar[7]; # trim velocity
αbar = xbar[8]; # trim angle of attack

 # Mach, dynamic pressure, static pressure from atmospheric values
Mach,qbar,ps = F16Model.atmos(h0,Vt0);
ρ = 2*qbar/Vt0^2; # calculating density in backwards way

# Matrix of what? TODO figure out what this matrix is doing
M = [cos(αbar) -Vbar*sin(αbar);
     sin(αbar)  Vbar*cos(αbar)];

# Sensor 1 -- Dynamic Pressure
C1 = [0 0 0 0 ρ*Vbar 0 0 ];

# Sensor 2,3 - Accelerometer
C23 = M*Al[5:6,:]; # need to change to corresponding values from Al and Bl
D23 = M*Bl[5:6,:];

# Sensor 4 -- Gyro
C4 = [0 0 0 0 0 0 1];

#sensor 5 - angle of attack sensor
#=
TODO figure out proper value... Assuming 1 for now...
most likely b/c sensor is directly measuring the state used in system
dynamic pressure had to convert pressure perturbation to velocity perturbation
=#
C5 = [0 0 0 0 0 1 0];

#=
 sensor 6 - gps sensor; Provides Npos and Epos
=#
C6 = [1 0 0 0 0 0 0]; # Npos
C7 = [0 1 0 0 0 0 0]; # Epos

#=
sensor 7 - barometer; Assume this outputs altitude directly
=#
C8 = [0 0 1 0 0 0 0]; # altitude

# Combined sensor y = C*x + D*d + n
C = [C6;C7;C8;C1;C23;C4;C5;];
#TODO understand why some are getting the D matrix while others are zero
D = [0;0;0;0;D23;0;0;];

# Add Filter to the disturbance
# F(s) = 1/(s/ωc + 1)
# (s/ωc + 1)do = di
# xfdot = -ωc*xf + ωc*di, do = xf
ωc = 1; #1 rads/
af = -ωc; bf = ωc; cf = 1;

# Augmented System Dynamics
Aa = [Al Bl; zeros(1,7) af]; #size changed b/c added Epos and Npos states
Ba = [zeros(7,1);bf];
Ca = [C D]; # Augmented measurement model to be used in Kalman filtering.

nz = size(Aa,1)
nu = size(Ba,2);

# Convert continuous time dynamics to discrete-time dynamics
S = [Aa Ba;
    zeros(nu,nz) zeros(nu,nu)];

dt = 1/50;
Md = exp(S*dt); # Matrix exponential.

# --- Discrete-time Dynamics ----
Ad = Md[1:nz,1:nz];
Bd =  Md[1:nz,(nz+1):end];

# Check stability
println("Absolute eigen values:\n", abs.(eigen(Ad).values))

#= Propagate uncertainty with AAd and BBd.
Define how disturbance is coming in.
Assume it is coming in from dele.
=#

d2r = pi/180;
#this increases to amount of states we have + 1 (added 2 from Npos and Epos)
xTrue = [0.,0.,0.,0.,0.,0.,0.,0.]; #TODO will Npos and Epos xtrue change over timesteps?

#=
ymeasured = g(x) + n
ybar + ytil = g(x + xtil)
ytil = C*xtil + n -- Model
ytil = ymeasured - ybar; the measurement we feed to Kalman filter.
ybar = g(xbar)
=#

##
μ_post = zeros(nz,1);
Σ_post = zeros(nz,nz);

μ_prior = zeros(nz,1);
Σ_prior = zeros(nz,nz);


 # aoa wil be on magnitude of +-0.08726656 rads (5deg) : |1%| is about .000873
# gps sensor is accurate to w/in 2.3ft TODO is this correct way of implementing?
#SSM need to be some small % of trim values (simulates small errors)
SensorScalingMatrix = Diagonal([2.3,         # Npos, Unsure about value used
                                2.3,         # Epos, Unsure about value used
                                5,           # altitude
                                2,           # qbar scaling
                                1,           # u acceleration
                                1,           # w acceleration
                                0.1,         # pitch rate, q?
                                0.000873,    # Angle of Attack

]);
DisturbanceScalingMatrix = sqrt(5);

Q = DisturbanceScalingMatrix*DisturbanceScalingMatrix'; #  degree^2 covariance
R = SensorScalingMatrix*SensorScalingMatrix'; # Noises of the sensors - diagonal matrix.

ny = size(C,1);

nSteps = 200; # Propagate nSteps
Sig = [];faultlist = []; machlist = []; altlist = [];
basefault = collect(range(0,1,length=nSteps)) .^ 2
for i in 1:nSteps
    μ_prior .= Ad*μ_post; # Propagate μ
    Σ_prior .= Ad*Σ_post*Ad' + Bd*Q*Bd'; # Propagate Σ
    # println(" Diagonal(Σprior): ", Diagonal(Σ_prior).diag);

    # Get sensor data
    n = SensorScalingMatrix*randn(ny); # E(n*n') := R
    d = DisturbanceScalingMatrix*rand(1); # E(d*d') := Q;

    # Simulating the measurement.
    y = Ca*xTrue + n; # Measurement data
    xTrue .= Ad*xTrue + Bd*d; # Dynamics of the true system

    # Compute Kalman gain
    K = Σ_prior*Ca'*inv(Ca*Σ_prior*Ca' + R);

    # Compute posterior with Kalman filteringqbar
    μ_post .= μ_prior + K*(y-Ca*μ_prior);
    Σ_post .= (I(nz) - K*Ca)*Σ_prior;

    #exact state
    altexact = (h0+xTrue[3])*0.3048  #converted ft to meters
    altexact_norm = (altexact - alt_mean)/alt_std #normalizing
    machexact,qbar,ps = F16Model.atmos(h0+xTrue[3],Vt0+xTrue[5]);
    machexact_norm = (machexact - mach_mean)/mach_std #normalizing
    #saving mach and alt steps for graphing later
    push!(machlist, machexact)
    push!(altlist, altexact)

    #kalman state
    altKF = (h0+μ_post[3])*0.3048  #converted feet to meters
    altKF_norm = (altKF - alt_mean)/alt_std  #normalizing
    machKF,qbar,ps = F16Model.atmos(h0+μ_post[3],Vt0+μ_post[5]);
    machKF_norm = (machKF - mach_mean)/mach_std  #normalizing

    #pressure prediction
    global fault = basefault[i]
    global fault_norm = (fault - fault_mean)/fault_std #set to desired fault
    Pstag_norm = pressuremodel([altexact_norm,machexact_norm,fault_norm])[1]
    #fault prediction
    faultpredict_norm = faultmodel([altKF_norm,machKF_norm,Pstag_norm])[1]
    faultpredict = (faultpredict_norm * fault_std) + fault_mean
    push!(faultlist, faultpredict)
    print(faultpredict,"\n")
    #print(faultpredict,"\n") #should desired fault parameter
    #print(μ_post,"\n")
    # No filtering.
    # μ_post = μ_prior;
    # Σ_post = Σ_prior;

    # Saving the data the open-loop uncertainty propagation.
    # print("μ: ",μ);
    # println(" Diagonal(Σ): ", Diagonal(Σ_post).diag);
    #push!(Sig,Diagonal(Σ_post).diag)
    #
    # # Save the  data with Kalman filtering switch ed on.
end

#TODO: Error Analysis do std and mean of error

#creating 3d line plot to display results
plot(machlist,altlist,faultlist,
        title="Fault Prediction over Mach/Altitude Flight Path",
        xlabel="mach",
        ylabel="alt (m)",
        zlabel="Fault Parameter",
        label="Neural Network",
        zlims=[-0.2,1],
        lw=2,
        camera=(15,30),
        legend=:bottomleft)   #actual results
plot!(machlist,altlist,basefault,
        label="Base Fault",
        l2=2)  #base values to compare

plot!(machlist,altlist,zeros(nSteps),
        label="Path Trace")  #trace on bottom

for i in 1:nSteps÷5
    plot!([machlist[5*i],machlist[5*i]],
            [altlist[5*i],altlist[5*i]],
            [basefault[5*i],faultlist[5*i]],
            linecolor=:grey,label=nothing)
    if i == 1
        plot!([machlist[5*i],machlist[5*i]],
                [altlist[5*i],altlist[5*i]],
                [basefault[5*i],faultlist[5*i]],
                linecolor=:grey,label="Difference")
    end
end
current()
