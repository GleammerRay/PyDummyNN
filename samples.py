from dummynn import Algo

def sampleOnly1s(hiddenCount : int = 5, iterationCount : int = 20, maxBias : float = 0.5, generationCount : int = 10000):
    def test(algo):
        _correct = 0
        _outputs = algo.fire([0], iterationCount)
        for output in _outputs:
            if (output == 1):
                _correct += 1
        _len = len(_outputs)
        if _len == 0:
            return 0
        return float(_correct) / _len

    algo = Algo.empty(1, hiddenCount, 2)
    score = test(algo)
    algo.resetNeuroneValues()
    result = algo.fire([0], iterationCount)
    num = 0
    for i in result:
        if i == 1:
            num += 1
    print('Before training:')
    print('Score: ' + str(score))
    print('Results:')
    print('% of 1s: ' + str(num / len(result) * 100) + '%')
    print('Training to get 1s...')
    algo = algo.trainRandom(test, maxBias, generationCount)
    score = test(algo)
    algo.resetNeuroneValues()
    result = algo.fire([0], iterationCount)
    num = 0
    for i in result:
        if i == 1:
            num += 1
    print('After training:')
    print('Results:')
    print('Score: ' + str(score))
    print('% of 1s: ' + str(num / len(result) * 100) + '%')
    return algo

def sample1sAnd0s(hiddenCount : int = 10, iterationCount : int = 20, maxBias : float = 0.5, generationCount : int = 10000):
    def test(algo):
        _correct0 = 0
        _outputs0 = algo.fire([0], iterationCount)
        for output in _outputs0:
            if (output == 0):
                _correct0 += 1
        _correct1 = 0
        algo.resetNeuroneValues()
        _outputs1 = algo.fire([1], iterationCount)
        for output in _outputs1:
            if (output == 1):
                _correct1 += 1
        _len0 = len(_outputs0)
        _len1 = len(_outputs1)
        if _correct0 == 0:
            return 0.0
        if _correct1 == 0:
            return 0.0
        if _len0 == 0:
            return 0.0
        if _len1 == 0:
            return 0.0
        return (float(_correct0) / float(_len0)) + (float(_correct1) / float(_len1))

    algo = Algo.empty(2, hiddenCount, 2)
    score = test(algo)
    algo.resetNeuroneValues()
    result0 = algo.fire([0], iterationCount)
    algo.resetNeuroneValues()
    result1 = algo.fire([1], iterationCount)
    num0 = 0
    for i in result0:
        if i == 1:
            num0 += 1
    num1 = 0
    for i in result1:
        if i == 1:
            num1 += 1
    print('Before training:')
    print('Score: ' + str(score))
    print('Results:')
    print('% of 0s: ' + str((num0 / len(result0)) * 100) + '%')
    print('% of 1s: ' + str((num1 / len(result1)) * 100) + '%')
    print('Training...')
    algo = algo.trainRandom(test, maxBias, generationCount)
    score = test(algo)
    algo.resetNeuroneValues()
    result0 = algo.fire([0], iterationCount)
    algo.resetNeuroneValues()    
    result1 = algo.fire([1], iterationCount)    
    num0 = 0
    for i in result0:
        if i == 0:
            num0 += 1
    num1 = 0
    for i in result1:
        if i == 1:
            num1 += 1
    print('After training:')
    print('Score: ' + str(score))
    print('Results:')
    print('% of 0s: ' + str((num0 / len(result0)) * 100) + '%')
    print('% of 1s: ' + str((num1 / len(result1)) * 100) + '%')
    return algo
