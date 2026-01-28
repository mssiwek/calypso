from ..emulator.emulator import Emulator
import numpy as np

class Inference:
    def __init__(self):
        self.emulator = Emulator()
        
    
    
    
    
    def likelihood(self, mdot_emulated, mdot_test) -> float:
      
      pass
    
    def loop_over_params(self, eb_values: np.ndarray, qb_values: np.ndarray) -> list:
      results = np.array((len(eb_values),len(qb_values)))
      for i,eb in enumerate(eb_values):
        for j,qb in enumerate(qb_values):
          emulator_output = self.emulator.predict(eb, qb)
          t, mdot = emulator_output['time'], emulator_output['mean']
          #calculate probability that the emulated mdot matches the test data
          results[i,j] = self.likelihood(t, mdot, eb, qb)
      return results
    
    def infer(self, t: np.ndarray, mdot: np.ndarray) -> dict:
        # Perform inference using the model
        
        return output
    