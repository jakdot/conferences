"""
Model replicating data from Lebiere.
"""

#TODO:
#    1. store chunk after done, even if production compilation
#    2. check RTs data
#    3. put results into quant modeling

import re
import math
import random
import simpy
import collections
import numpy as np
import pandas as pd
import sys

import pyactr as actr
import pyactr.utilities as utilities

DECAY = 0.5

#MP = 1.3
#RT = 2.8

SIMULATIONNUMBER = int(sys.argv[1])
#MP = float(sys.argv[2])
#RT = float(sys.argv[3])

MP = 2.0
RT = 4

ADDITION = str(SIMULATIONNUMBER) + str(MP) + str(RT)

addition = actr.ACTRModel(subsymbolic=True, retrieval_threshold=-2.25, latency_factor=0.1, latency_exponent=0.5, instantaneous_noise = 0.25, decay=DECAY, buffer_spreading_activation={"g": 1}, strength_of_association=4, association_only_from_chunks=False)

actr.chunktype("countOrder", ("first", "second"))

actr.chunktype("number", "value")

actr.chunktype("add", ("state", "arg1", "arg2", "sum"))

dm = addition.decmem

numbers = []

for i in range(0, 20):
    #numbers.append(actr.makechunk("number"+str(i), "number", value=i))
    numbers.append(str(i))

#for i in range(0, 16):
#    dm.add(actr.makechunk("chunk"+str(i), "countOrder", first=numbers[i], second=numbers[i+1]))

SEC_IN_YEAR = 365*24*3600

actr.chunktype("add", ("state", "arg1", "arg2", "sum"))

NUMBER = 2000 #number of iterations

TIME = 20*86400 #time span in which training occurs

UPTO = 9 #upto which number is counted

SIMULATION = 1000 #number of steps in each simulation cycle

PROB = [[(2 - i/UPTO)*(2 - j/UPTO) for j in range(1, UPTO+1)] for i in range(1, UPTO+1)]
summing = sum(sum(x) for x in PROB)
PROB = [[x/summing for x in y] for y in PROB]
summing = sum(sum(x) for x in PROB)
LEMMA_CHUNKS = {}
for i in range(UPTO):
    for j in range(UPTO):
        count_freq = PROB[i][j] * NUMBER
        time_interval = TIME / count_freq
        chunk_times = np.arange(start=-time_interval, stop=-(time_interval*count_freq)-1, step=-time_interval)
        LEMMA_CHUNKS[actr.makechunk("", typename="add", arg1=numbers[i+1], arg2=numbers[j+1], sum=i+1+j+1)] = math.log(np.sum((0.2-chunk_times) ** (-DECAY)))

addition.set_decmem({x: np.array([]) for x in LEMMA_CHUNKS})

addition.decmem.activations.update(LEMMA_CHUNKS)

for x in range(1,UPTO+1):
    for y in range(1, UPTO+1):
        print(x, y, addition.decmem.activations[actr.makechunk("", typename="add", arg1=numbers[x], arg2=numbers[y], sum=x+y)])

addition.productionstring(name="retrieve_addition", string="""
        =g>
        isa     add
        state   reading
        arg1    =num1
        arg2    =num2
        sum     None
        ?retrieval>
        state   free
        buffer  empty
        ==>
        =g>
        isa     add
        state   None
        +retrieval>
        isa     add
        arg1     =num1
        arg2     =num2""")

addition.productionstring(name="terminate_addition", string="""
        =g>
        isa     add
        state   None
        =retrieval>
        isa     add
        arg1    ~None
        sum     =v7
        ==>
        ~retrieval>
        =g>
        isa     add
        sum     =v7""")

addition.productionstring(name="done", string="""
        =g>
        isa     add
        state   None
        sum     =v7
        sum     ~None
        ==>
        =g>
        isa     add
        state   None""")


results = {}

attempts = collections.Counter()

#for i in range(0, 16):
#    dm[actr.makechunk("chunk"+str(i), "countOrder", first=numbers[i], second=numbers[i+1])] = np.array([0])

DELAY = 7500 #delay in simulation
init_time = 0

for i in range(len(numbers)-1):
    addition.set_similarities(numbers[i], numbers[i+1], -MP+(i/(i+1))**2)

#values from p. 24

#for i in range(len(numbers)):
#    for j in range(len(numbers)-1):
#        addition.set_similarities(numbers[i], numbers[j], -abs(0.15*(i-j)))
#addition.model_parameters["retrieval_threshold"] = -2.25

#addition.model_parameters["activation_trace"] = True
addition.model_parameters["partial_matching"] = True
addition.model_parameters["retrieval_threshold"] = -RT
addition.model_parameters["mismatch_penalty"] = MP

PROB2 = [sum(x) for x in PROB] #used for random sample

PROB2 = [sum(PROB2[0:i]) for i in range(len(PROB2))]

PROB2.append(1)

def simulation2(start, end, init_time, results, timing, time_stamps, feedback=True, learning=False, ul=None):
    """
    Second simulation, using production learning.
    """

    attempts = collections.Counter()

    if learning:

        addition.productionstring(name="retrieve_addition", string="""
        =g>
        isa     add
        state   reading
        arg1    =num1
        arg2    =num2
        sum     None
        ?retrieval>
        state   free
        buffer  empty
        ==>
        =g>
        isa     add
        state   None
        +retrieval>
        isa     add
        arg1     =num1
        arg2     =num2""", utility=ul[0])

        addition.productionstring(name="terminate_addition", string="""
        =g>
        isa     add
        state   None
        =retrieval>
        isa     add
        arg1    ~None
        sum     =v7
        ==>
        ~retrieval>
        =g>
        isa     add
        sum     =v7""", utility=ul[1])

        addition.productionstring(name="done", string="""
        =g>
        isa     add
        state   None
        sum     =v7
        sum     ~None
        ==>
        =g>
        isa     add
        state   None""", reward=15)

        addition.model_parameters["production_compilation"] = True
        addition.model_parameters["utility_learning"] = True
        addition.model_parameters["utility_noise"] = 0.7

    for loop in range(start+1,end+1):
        x = random.uniform(0, 1) #what problem should be used 
        for i in range(len(PROB2)):
            if x < PROB2[i]:
                break
        else:
            i = UPTO
        x = random.uniform(0, 1) #what problem should be used 
        for j in range(len(PROB2)):
            if x < PROB2[j]:
                break
        else:
            j = UPTO
        #print(i, j)
        correct = i + j
        attempts.update([(i, j)])
        addition.goal.add(actr.makechunk("", "add", state="reading", arg1=numbers[i], arg2=numbers[j]))

        sim = addition.simulation(initial_time=init_time, trace=False, gui=False)
        addition.used_productions.rules.used_rulenames = {}
        while True:
            try:
                sim.step()
                #print(sim.current_event)
            except simpy.core.EmptySchedule:
                break
            if re.search("RETRIEVED: None", sim.current_event.action):
                results.setdefault((str(i), str(j)), collections.Counter()).update('X')
                if feedback and random.randint(0,1) == 1:
                    addition.retrieval.add(actr.makechunk("", "add", arg1=numbers[i], arg2=numbers[j], sum=numbers[correct]))
                    addition.retrieval.clear(sim.show_time()) #advise -- teacher tells the correct answer
                break
            if re.search("RULE FIRED: done", sim.current_event.action):
                x = addition.goal.copy().pop()
                addition.goal.clear(sim.show_time())
                x = x._asdict()
                results.setdefault((str(i), str(j)), collections.Counter()).update([str(x["sum"])])
                if str(x["sum"]) != str(numbers[correct]):
                    if feedback and random.randint(0,1) == 1:
                        addition.retrieval.add(actr.makechunk("", "add", arg1=numbers[i], arg2=numbers[j], sum=numbers[correct]))
                        addition.retrieval.clear(sim.show_time()) #advise -- teacher tells the correct answer in 50% of cases if incorrect
                        if learning:
                            utilities.modify_utilities(sim.show_time(), -15, addition.used_productions.rules.used_rulenames, addition.used_productions.rules, addition.used_productions.model_parameters)
                elif timing:
                    time_stamps.setdefault((str(i), str(j)), []).append(sim.show_time() - init_time)
                break
        addition.retrieval.state = addition.retrieval._FREE
        addition.retrieval.clear()
        init_time = loop*DELAY
        #input()
    return results, time_stamps, init_time, ul

def record(results, simulation, output="output"):

    simulation = str(simulation)

    for i in range(1, UPTO+1):
        temp_l.append(str(i))
        summed = 0
        correct_summed = 0
        for j in range(1, UPTO+1):
            val = results.get((str(i), str(j)))
            print(i, j, val)
            if val:
                summed += sum(val.values())
                correct_summed += val.setdefault(str(i+j), 0)
        temp_s.append(summed)
        temp_c.append(correct_summed)
        temp_sim.append(simulation)
        try:
            temp_r.append(correct_summed/summed)
        except ZeroDivisionError:
            temp_r.append(0)

    sample = pd.DataFrame.from_dict({"Augend": temp_l, "Correct": temp_c, "Summed": temp_s, "Ratio": temp_r, "Simulation": temp_sim})
    sample.to_csv(output + ADDITION + ".csv", sep=",", encoding="utf-8")
    print("DONE RECORDING")
    return results

def record_time_stamps(time_stamps, simulation, output="output_time_stamps"):

    for i in range(1, UPTO+1):
        for j in range(1, UPTO+1):
            temp_l1.append(str(i))
            temp_l2.append(str(j))
            temp_l12.append(str(i+j))
            t = time_stamps.get((str(i), str(j)))
            if t:
                temp_t.append(sum(t)/len(t))
            else:
                temp_t.append(0)

    sample = pd.DataFrame.from_dict({"Augend": temp_l1, "Addend": temp_l2, "Sum": temp_l12, "Time": temp_t, "Simulation": [simulation]*len(temp_l1)})
    sample.to_csv(output + ADDITION + ".csv", sep=",", encoding="utf-8")
    print("DONE RECORDING")
    return results

temp_l = []
temp_s = []
temp_c = []
temp_r = []
temp_sim = []
temp_l1 = []
temp_l2 = []
temp_l12 = []
temp_t = []

results, _, init_time, ul = simulation2(1, SIMULATION, init_time, results, False, {}, feedback=True, learning=True, ul=[5,5])

record(results, 1)

results = {}
results, time_stamps, init_time, ul = simulation2(SIMULATION*1, SIMULATION*2, init_time, results, True, {}, feedback=False, learning=False, ul=ul)
record_time_stamps(time_stamps, 6)
record(results, 6)

print(addition._ACTRModel__productions)

for x in range(1,UPTO+1):
    for y in range(1, UPTO+1):
        print(x, y, addition.decmem.activations[actr.makechunk("", typename="add", arg1=numbers[x], arg2=numbers[y], sum=x+y)])
