import numpy as np
from astropy import units as u
from astropy.constants import g0 # 9.81 m/s
from astropy.table import QTable

class StdAtm:
    """
    Class to calculate 
    """
    
    def __init__(self):
        self.height_boundaries = (0, 11000, 20000, 32000, 47000, 51000, 71000, 84852)*u.m
        self.tempK = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95)*u.K
        self.lapseRate = (-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002, 0)*u.K/u.m
        self.pressure = (101325, 22625.79, 5471.935, 867.255, 110.76658, 66.848296, 3.94900, 0.372657)*u.Pa
        self.R = 287*u.J/u.K/u.kg # specific gas constant for air
        self.gamma = 1.4 # heat capacity ratio (adiabatic index)
        
    def findBase(self, height):
        """Takes a single height, returns index to various self.xxx arrays"""
        baseHeights = self.height_boundaries
        for i in range(len(baseHeights)):
            if (baseHeights[i] > height): return i-1
        return i
        
    def findBases(self, heights):
        """
        heights can be single quantity or array
        TODO - vectorize the array part
        sets self.bases with array indices and self.is_isothermal with booleans
        """
        try:
            self.bases = [self.findBase(h) for h in heights]
            self.heights = heights
        except:
            self.bases = [self.findBase(heights), ]
            self.heights = [heights, ]
#         print(self.bases)
        self.is_isothermal = self.lapseRate[self.bases] == 0
        
    def calcTemp(self):
        return self.tempK[self.bases] + self.lapseRate[self.bases] \
                * (self.heights - self.height_boundaries[self.bases])
    
    def calcPress(self):
        def isothermal(inx, base_index):
            exponent = -g0 / (self.tempK[base_index] * self.R) * \
                        (self.heights[inx] - self.height_boundaries[base_index])
            return self.pressure[base_index] * np.exp(exponent)

        def nonisothermal(inx, base_index):
            exponent = -g0 / (self.lapseRate[base_index] * self.R) 
            return self.pressure[base_index] * (self.temp[inx] / self.tempK[base_index])**exponent
        
        P = np.zeros(self.heights.shape)*u.Pa
        for i in range(len(self.heights)):
            if self.is_isothermal[i]:
                P[i] = isothermal(i, self.bases[i])
            else:
                P[i] = nonisothermal(i, self.bases[i])
        return P
#         return [isothermal(i, self.bases[i]) if self.is_isothermal[i] else 
#                          nonisothermal(i, self.bases[i]) for i in range(len(self.heights))]
    
    def calcDensity(self, P):
        "parameter: pressure(s)"
        return (P / (self.R*self.temp)).to(u.kg*u.m**-3)
        
    def calcMach1(self, temp):
        # speed of sound
        return np.sqrt(self.gamma * self.R * temp)
    
    def calcResults(self, height, data):
        "data can be dictionary or astropy table, the syntax is identical"
        K2C = 273.15
        data["height"] = height.to(u.m)
        self.findBases(height)
        self.temp = self.calcTemp()
        data["temp_K"] = self.temp
        data["temp_C"] = (self.temp.value - K2C)*u.deg_C
        data["pressure"] = self.calcPress()
        data["rho"] = self.calcDensity(data["pressure"]) #.to(u.kg*u.m**-3)
        data["isothermal"] = self.is_isothermal
        data["mach1_si"] = self.calcMach1(data["temp_K"]).to(u.m/u.s)
        data["mach1_kt"] = data["mach1_si"] / 0.51444
        return data    
        
    
    def getResultsDict(self, height):
        """
        Requires: height(s) as Quantity with units, single or iterable.
        Returns a dictionary
        """
        data = {}
        return self.calcResults(height, data)
    
    def getResultsQtable(self, height):
        """
        Requires: height(s) as Quantity with units, single or iterable.
        Returns a QTable
        """
        data = QTable()
        return self.calcResults(height, data)