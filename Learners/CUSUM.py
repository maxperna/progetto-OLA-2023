class CUSUM:

  def __init__(self, N, eps, threshold):

    # number of samples needed to initialize the reference value:
    self.N = N

    # reference value:
    self.reference = 0

    # epsilon value: 
    self.eps = eps

    # threshold:
    self.threshold = threshold

    # g values:
    self.g_plus = 0
    self.g_minus = 0

    # number of rounds executed:
    self.t = 0


  def update(self, sample, pulled_arm):
    self.t += 1 # TODO: verificare se va messo a fine funzione

    if self.t <= self.N:
      self.reference += sample/self.N
      #if(pulled_arm == 2):
      #print("new arm REFERENCE: ", self.reference, "Sample:", sample)
      return False
    
    else:
      self.reference = (self.reference*(self.t-1) + sample)/self.t
      s_plus = (sample - self.reference) - self.eps
      s_minus = -(sample - self.reference) - self.eps
      self.g_plus = max(0, self.g_plus + s_plus)
      self.g_minus = max(0, self.g_minus + s_minus)

      # if(pulled_arm == 2):
      # print("time: ", self.t, "REFERENCE: ", self.reference, "Sample: ", sample)
      # print("s_plus: ", s_plus, "s_minus: ", s_minus, "g_plus: ", self.g_plus, "g_minus: ", self.g_minus)
      # print('')
      
      if self.g_plus > self.threshold or self.g_minus > self.threshold:
        print("Abrupt change detected!")
        self.reset()
        return True
      return False
    

  def reset(self):
    self.t = 0
    self.g_plus = 0
    self.g_minus = 0
    self.reference = 0