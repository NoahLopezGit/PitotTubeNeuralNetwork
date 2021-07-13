# PitotTubeNeuralNetwork
A Neural Network to characterize faults in Pitot-Tubes (as a sort of digital twin). <br />
use $ git clone https://github.com/FrostyNip/PitotTubeNeuralNetwork to clone this project on your machine

What each .jl file does:

**FaultPredictor.jl** 
- Constructs and trains fault detection network
 
**FaultNetworkTesting.jl** 
- tests network accuracy and plots some results
  
**PressurePredictor.jl** 
- Constructs and trains pressure prediction network
  
**PressureNetworkTesting.jl** 
- tests network accuracy and plots some results
  
**PitotSystem.jl** 
- Combines networks, f16 model, and kalman filtering to create system described in project architecture
  
**SIMSCALE_results_plotting.jl** 
- PLots some restuls from the CFD analysis
