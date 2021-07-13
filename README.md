# PitotTubeNeuralNetwork
A Neural Network to characterize faults in Pitot-Tubes (as a sort of digital twin).
use $ git clone https://github.com/FrostyNip/PitotTubeNeuralNetwork to clone this project on your machine

What each .jl file does:

FaultPredictor.jl \n
Constructs and trains fault detection network
 
FaultNetworkTesting.jl \n
tests network accuracy and plots some results
  
PressurePredictor.jl \n
Constructs and trains pressure prediction network
  
PressureNetworkTesting.jl \n
tests network accuracy and plots some results
  
PitotSystem.jl \n
Combines networks, f16 model, and kalman filtering to create system described in project architecture
  
SIMSCALE_results_plotting.jl \n
PLots some restuls from the CFD analysis
