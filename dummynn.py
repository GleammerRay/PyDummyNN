from copy import copy, deepcopy
from json import JSONEncoder
from random import randint, random, uniform
from types import NoneType

class Neurone:
    def __init__(self, connections : dict | NoneType = None):
        self._connections = connections if (connections is dict) else {}
        self.onFire = None
        self._value = 0

    @staticmethod
    def fromJSON(json : list, neurones : list):
        connections = {}
        for i in range(0, len(json)):
            connections[neurones[i]] = json[i]
        return Neurone(connections)

    def toString(self) -> str:
        result = ''
        i = 0
        for k, v in self._connections.items():
            i += 1
            if i == len(self._connections):
                result += str(v)
                break
            result += str(v) + '\n'
        result
        return result

    def toJSON(self) -> list:
        json = []
        for k, v in self._connections.items():
            json.append(v)
        return json

    def resetValue(self):
        self._value = 0.0
        
    def randomBias(self, maxBias : float):
        for k, v in self._connections.items():
            posi = -1 if randint(0, 1) == 0 else 1
            v = v + (uniform(0.0, maxBias) * posi)
            if (v < 0):
                v = 0
            if (v > 1):
                v = 1
            self._connections[k] = v
    
    def connect(self, neurone, value):
        self._connections[neurone] = value

    def fire(self):
        for k, v in self._connections.items():
            k.pulse(v)
        if (callable(self.onFire)):
            self.onFire()
    
    def pulse(self, value):
        self._value += value
        if (self._value < 0.0):
            self._value = 0.0
        if (self._value >= 1.0):            
            self._value = 0.0
            self.fire()

class Algo:
    def __init__(self, inputCount : int, hiddenCount : int, outputCount : int):        
        self._outputs = []
        for i in range(0, outputCount):
            neurone = Neurone()
            self._outputs.append(neurone)
        self._hidden = []
        for i in range(0, hiddenCount):
            neurone = Neurone()
            for i2 in range(0, len(self._outputs)):
                neurone.connect(self._outputs[i2], 0.5)
            self._hidden.append(neurone)
        self._inputs = []
        for i in range(0, inputCount):
            neurone = Neurone()
            for i2 in range(0, hiddenCount):
                neurone.connect(self._hidden[i2], 0.5)
            self._inputs.append(neurone)

    def toString(self) -> str:
        result = '[inputs]'        
        for neurone in self._inputs:
            result += '\n' + '---'
            result += '\n' + neurone.toString()
        result += '\n[hidden]'
        for neurone in self._hidden:
            result += '\n' + '---'
            result += '\n' + neurone.toString()
        return result

    def toJSON(self) -> dict:
        json = { 'inputs': [], 'hidden': [] }
        for neurone in self._inputs:
            json['inputs'].append(neurone.toJSON())
        for neurone in self._hidden:
            json['hidden'].append(neurone.toJSON())
        return json

    def randomBias(self, maxBias : float):
        for neurone in self._inputs:
            neurone.randomBias(maxBias)
        for neurone in self._hidden:
            neurone.randomBias(maxBias)

    def resetNeuroneValues(self):
        for neurone in self._inputs:
            neurone.resetValue()
        for neurone in self._hidden:
            neurone.resetValue()
        for neurone in self._outputs:
            neurone.resetValue()

    def fire(self, inputs : list, iterationCount : int) -> list:
        _results = []
        for i in range(0, len(self._outputs)):
            def _loop():
                a = i
                def _appendResult():
                    _results.append(a)
                self._outputs[i].onFire = _appendResult
            _loop()
        for i in range(0, iterationCount):
            for _input in inputs:
                self._inputs[_input].fire()
        return _results

    def trainRandom(self, test, maxBias : float, generationCount : int):
        _bestAlgo = deepcopy(self)
        _bestAlgo.resetNeuroneValues()
        _bestAlgoScore = test(_bestAlgo)
        _bestAlgo.resetNeuroneValues()
        for i in range(0, generationCount):
            _curAlgo = deepcopy(_bestAlgo)
            _curAlgo.randomBias(maxBias)
            _score = test(_curAlgo)
            if _score > _bestAlgoScore:
                _bestAlgo = _curAlgo
                _bestAlgoScore = _score
        _bestAlgo.resetNeuroneValues()
        return _bestAlgo