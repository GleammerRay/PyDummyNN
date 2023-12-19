from copy import deepcopy
from random import randint, uniform
from types import NoneType

class Neurone:
    def __init__(self, connections : dict | NoneType = None, value : float | NoneType = None, minValue : float | NoneType = None, maxValue : float | NoneType = None):
        self._connections = {} if (connections is None) else connections
        self.onFire = None
        self._value = 0.0 if (value is None) else value
        self._initValue = self._value
        self._minValue = 0.0 if (minValue is None) else minValue
        self._maxValue = 1.0 if (maxValue is None) else maxValue

    @staticmethod
    def fromJSON(json : dict, connectedNeurones : list | NoneType = None):
        _jsonConnections = json['connections']
        connections = {}
        if connectedNeurones is not None:
            for i in range(0, len(connectedNeurones)):
                connections[connectedNeurones[i]] = _jsonConnections[i]
        return Neurone(connections, json['initValue'], json['minValue'], json['maxValue'])

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

    def toJSON(self) -> dict:
        json = {
            'connections': None if (len(self._connections) == 0) else list(self._connections.values()),
            'initValue': self._initValue,
            'minValue': self._minValue,
            'maxValue': self._maxValue
        }
        return json

    def resetValue(self):
        self._value = self._initValue
        
    def randomBias(self, maxBias : float):
        for k, v in self._connections.items():
            posi = -1 if randint(0, 1) == 0 else 1
            v = v + (uniform(0.0, maxBias) * posi)
            if (v < self._minValue):
                v = self._minValue
            if (v > self._maxValue):
                v = self._maxValue
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
        if (self._value < self._minValue):
            self._value = self._minValue
        if (self._value >= self._maxValue):            
            self._value = self._initValue
            self.fire()

class Algo:
    def __init__(self, inputs : list, hidden : list, outputs : list):
        self._inputs = inputs
        self._hidden = hidden
        self._outputs = outputs

    @staticmethod
    def empty(inputCount : int, hiddenCount : int, outputCount : int, value : float | NoneType = None, minValue : float | NoneType = None, maxValue : float | NoneType = None):
        _outputs = []
        for i in range(0, outputCount):
            neurone = Neurone(None, value, minValue, maxValue)
            _outputs.append(neurone)
        _hidden = []
        for i in range(0, hiddenCount):
            neurone = Neurone(None, value, minValue, maxValue)
            for i2 in range(0, len(_outputs)):
                neurone.connect(_outputs[i2], 0.5)
            _hidden.append(neurone)
        _inputs = []
        for i in range(0, inputCount):
            neurone = Neurone(None, value, minValue, maxValue)
            for i2 in range(0, hiddenCount):
                neurone.connect(_hidden[i2], 0.5)
            _inputs.append(neurone)
        return Algo(_inputs, _hidden, _outputs)

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

    @staticmethod
    def fromJSON(json : dict):
        _outputs = []
        for _output in json['outputs']:
            _outputs.append(Neurone.fromJSON(_output))
        _hiddenList = []
        for _hidden in json['hidden']:
            _hiddenList.append(Neurone.fromJSON(_hidden, _outputs))
        _inputs = []
        for _input in json['inputs']:
            _inputs.append(Neurone.fromJSON(_input, _hiddenList))
        return Algo(_inputs, _hiddenList, _outputs)

    def toJSON(self) -> dict:
        json = { 'inputs': [], 'hidden': [], 'outputs': [] }
        for neurone in self._inputs:
            json['inputs'].append(neurone.toJSON())
        for neurone in self._hidden:
            json['hidden'].append(neurone.toJSON())
        for neurone in self._outputs:
            json['outputs'].append(neurone.toJSON())
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

    def pulse(self, inputs : dict, iterationCount : int) -> list:
        _results = []
        for i in range(0, len(self._outputs)):
            def _loop():
                a = i
                def _appendResult():
                    _results.append(a)
                self._outputs[i].onFire = _appendResult
            _loop()
        for i in range(0, iterationCount):
            for k, v in inputs.items():
                self._inputs[k].pulse(v)
        return _results

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