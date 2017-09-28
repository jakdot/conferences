"""
Model simulating data from Zajenkowski, Styla and Szymanik, 2011 -- verification of proportional quantifiers.
"""

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
import string
import warnings

DECAY = 0.5

#MP = 1.3
#RT = 2.8

SIMULATIONNUMBER = int(sys.argv[1])
MP = 2.0
RT = 4

addition = actr.ACTRModel(subsymbolic=True, retrieval_threshold=-2.25, latency_factor=0.1, latency_exponent=0.5, instantaneous_noise = 0.25, decay=DECAY, buffer_spreading_activation={"g": 1}, strength_of_association=4, association_only_from_chunks=False)

actr.chunktype("countOrder", ("first", "second"))

actr.chunktype("number", "value")

actr.chunktype("add", ("state", "arg1", "arg2", "sum"))

dm = addition.decmem

numbers = []

for i in range(0, 30):
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
print(PROB)
for i in range(UPTO):
    for j in range(UPTO):
        count_freq = PROB[i][j] * NUMBER
        time_interval = TIME / count_freq
        chunk_times = np.arange(start=-time_interval, stop=-(time_interval*count_freq)-1, step=-time_interval)
        LEMMA_CHUNKS[actr.makechunk("", typename="add", arg1=numbers[i+1], arg2=numbers[j+1], sum=i+1+j+1)] = math.log(np.sum((0.2-chunk_times) ** (-DECAY)))

LEMMA_CHUNKS[actr.makechunk("", typename="add", arg1=numbers[11], arg2=numbers[4], sum=numbers[15])] = -3.5
LEMMA_CHUNKS[actr.makechunk("", typename="add", arg1=numbers[13], arg2=numbers[2], sum=numbers[15])] = -3.5 #to be found by simulation

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

def simulation2(start, end, init_time, results, timing, time_stamps, feedback=True, learning=False):
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
        arg2     =num2""", utility=1)

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
        sum     =v7""", utility=1)

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
        addition.model_parameters["utility_noise"] = 0.8

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
                        addition.retrieval.clear(sim.show_time()) #advise -- teacher tells the correct answer in roughly 50% of cases if incorrect
                        if learning:
                            utilities.modify_utilities(sim.show_time(), -15, addition.used_productions.rules.used_rulenames, addition.used_productions.rules, addition.used_productions.model_parameters)
                elif timing:
                    time_stamps.setdefault((str(i), str(j)), []).append(sim.show_time() - init_time)
                break
        addition.retrieval.state = addition.retrieval._FREE
        addition.retrieval.clear()
        init_time = loop*DELAY
    return results, time_stamps, init_time

_, _, init_time = simulation2(1, SIMULATION, init_time, results, False, {}, feedback=True, learning=True)

class Model(object):
    """
    Model searching and attending to various stimuli.
    """

    def __init__(self, env, **kwargs):
        self.m = actr.ACTRModel(environment=env, **kwargs)

        actr.chunktype("pair", "probe answer")
        
        actr.chunktype("goal", "state, arg1, arg2, sum, end, value")

        self.dm = self.m.decmem

        self.start = actr.makechunk(nameofchunk="start", typename="chunk", value="start")
        actr.makechunk(nameofchunk="done", typename="chunk", value="done")
        self.m.set_goal("g2")
        self.m.goals["g2"].delay=0.2

        self.m.productionstring(name="find_probe", string="""
        =g>
        isa     goal
        state   start
        value   =vv
        ?visual_location>
        buffer  empty
        ==>
        =g>
        isa     goal
        state   attending
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        value   =vv
        screen_x lowest
        screen_y closest""") #this rule is used if automatic visual search does not put anything in the buffer
        
        self.m.productionstring(name="move_in_line", string="""
        =g>
        isa     goal
        state   attending
        value   =vv
        sum     ~None
        ?visual_location>
        state   error
        ==>
        =g>
        isa     goal
        state   attending
        sum     None
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        value   =vv
        screen_x lowest
        screen_y closest""", utility=20) #this rule is used if automatic visual search does not put anything in the buffer

        self.m.productionstring(name="check_probe", string="""
        =g>
        isa     goal
        state   start
        ?visual_location>
        buffer  full
        ==>
        =g>
        isa     goal
        state   attending""")  #this rule is used if automatic visual search is enabled and it puts something in the buffer

        self.m.productionstring(name="attend_probe", string="""
        =g>
        isa     goal
        state   attending
        =visual_location>
        isa    _visuallocation
        screen_x    ~None
        ?visual>
        state   free
        ==>
        =g>
        isa     goal
        state   reading
        sum     None
        +visual>
        isa     _visual
        cmd     move_attention
        screen_pos =visual_location""")

        self.m.productionstring(name="encode_probe", string="""
        =g>
        isa     goal
        state   reading
        =visual>
        isa     _visual
        value   =val
        ==>
        =g>
        isa     goal
        sum     None
        state   None""")

        self.m.productionstring(name="retrieve_addition", string="""
        =g>
        isa     add
        state   None
        arg1    =num1
        arg2    =num2
        arg2    ~None
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
        arg2     =num2""", utility=7)

        self.m.productionstring(name="terminate_addition", string="""
        =g>
        isa     add
        =retrieval>
        isa     add
        arg1    ~None
        sum     =v7
        ==>
        ~retrieval>
        =g>
        isa     add
        state   move
        sum     =v7""", utility=7)

        for i, nums in enumerate([(2, 2), (4, 4), (4, 2), (3, 2), (2, 3), (5, 2), (7, 2), (7, 4), (9, 2), (9, 4), (11, 4), (13, 2)]):
            if nums[0] + nums[1] < 10:
                util = 7.5
            else:
                util = 7
            self.m.productionstring(name="retrieve_addition_terminate_addition" + str(i), string="""
        =g>
        isa     add
        state   None
        arg1    """+str(nums[0])+"""
        arg2    """+str(nums[1])+"""
        sum     None
        ?retrieval>
        state   free
        buffer  empty
        ==>
        =g>
        isa     add
        state   move
        sum     """+str(nums[0]+nums[1]), utility=util)

        self.m.productionstring(name="retrieve failure", string="""
        =g>
        isa     goal
        state   None
        ?retrieval>
        state   error
        ==>
        =g>
        isa     goal
        state   None
        arg2    None""")
        
        for i in range(15):
            self.m.productionstring(name="fast_increment" + str(i), string="""
        =g>
        isa     goal
        state   None
        arg1   """+str(i)+"""
        sum     None
        ?retrieval>
        state   free
        buffer  empty
        ==>
        =g>
        isa     goal
        state   move
        sum   """+str(i+1), utility=6)

        self.m.productionstring(name="move_vis_loc", string="""
    =g>
    isa     goal
    state   move
    sum     =csum
    sum     ~None
    value   =vv
    =visual_location>
    isa     _visuallocation
    screen_y    =sy
    ==>
    =g>
    isa     goal
    state   attending
    arg2    None
    arg1    =csum
    ~visual>
    ?visual_location>
    attended False
    +visual_location>
    isa _visuallocation
    value   =vv
    screen_x closest
    screen_y    =sy""")

        self.m.productionstring(name="can_find", string="""
    =g>
    isa     goal
    sum         =x
    end         =x
    ?manual>
    state   free
    ==>
    ~retrieval>
    ~g>
    +manual>
    isa     _manual
    cmd     press_key
    key     'J'""", utility=5)

        self.m.productionstring(name="cannot_find", string="""
    =g>
    isa     goal
    state   attending
    end     ~unknown
    ?visual_location>
    state   error
    ?manual>
    state   free
    ==>
    ~retrieval>
    ~g>
    +manual>
    isa     _manual
    cmd     press_key
    key     'F'""", utility=-5)
        
        self.m.productionstring(name="switch", string="""
    =g>
    isa     goal
    state   attending
    end     unknown
    sum     None
    ?visual_location>
    state   error
    ?manual>
    state   free
    ==>
    =g>
    isa     goal
    state   attending
    value   B
    end     15
    ?visual_location>
    attended False
    +visual_location>
    isa _visuallocation
    value   B
    screen_x lowest
    screen_y closest""", utility=20)

        self.productions = self.m._ACTRModel__productions

groups = ['WBWWB', 'BBWW', 'WBBBBW']
numbers_c = {"B": [], "W": []}
for group in groups:
    numbers_c["B"].append(group.count("B"))
    numbers_c["W"].append(group.count("W"))
pos_bw = {"B": ['120', '100', '120'], "W": ['100', '140', '100']}
all_pos = {"B": {0: [], '120': [2, 5], "170": [11, 12], "220": [22, 23, 24, 25]}, "W": {0: [], '120': [1, 3, 4], "170": [13, 14], "220": [21, 26]}}

stim_d = {}
for pos, group in enumerate(groups):
    print(pos)
    stim_d.update({(pos*10)+(key+1): {'text': letter, 'position': (100+key*20, 120+50*pos)} for key, letter in enumerate(group)})
environ = actr.Environment()
m = Model(environ, subsymbolic=True, latency_factor=0.1, decay=0.5, retrieval_threshold=-10, instantaneous_noise=0.1, automatic_visual_search=False, eye_mvt_scaling_parameter=0.1, eye_mvt_angle_parameter=0.1, utility_noise=0.7) 
times = []
for loop in range(100):

    environ.current_focus = (80, 120)
    
    m.m.goals["g"].add(actr.makechunk(typename="read", state=m.start, arg1='0', arg2=None, end="unknown", value='W'))

    m.m.decmems = {}
    m.m.set_decmem({x: np.array([]) for x in LEMMA_CHUNKS})

    m.m.decmem.activations.update(LEMMA_CHUNKS)
    m.m.set_retrieval("retrieval")
    m.m.visbuffers = {}
    m.m.visualBuffer("visual", "visual_location", m.m.decmem, finst=15)

    sim = m.m.simulation(realtime=False, trace=False, gui=True, initial_time=init_time+2.0, environment_process=environ.environment_process, stimuli=stim_d, triggers='J', times=15)
    while True:
            try:
                sim.step()
            except simpy.core.EmptySchedule:
                break
            #print(sim.current_event)
            if sim.show_time() - init_time > 15:
                break
            #print(sim.current_event)
            #print(m.m.goals["g"])
            if m.m.visbuffers["visual"] and m.m.goals["g"] and (list(m.m.goals["g"])[0].arg2 == "None" or list(m.m.goals["g"])[0].arg2 == None) and (list(m.m.goals["g"])[0].state != "move") and (list(m.m.goals["g"])[0].state != "attending"):
                vis = m.m.visbuffers["visual"]._data.copy().pop()._asdict()
                vl = m.m.visbuffers["visual_location"]._data.copy().pop()._asdict()
                #print(vis)
                #print(vl)
                if vis["screen_pos"]._asdict()["screen_y"] == vl["screen_y"] and vis["screen_pos"]._asdict()["screen_x"] == vl["screen_x"]:
                    y = vis["screen_pos"]._asdict()["screen_y"]
                    x = vis["screen_pos"]._asdict()["screen_x"]
                    letter = str(vis["value"])
                    val = 0
                    if y == '120' and x == pos_bw[letter][0]:
                        val = numbers_c[letter][0]
                    elif y == '170'and x == pos_bw[letter][1]:
                        val = numbers_c[letter][1]
                    elif y == '220' and x == pos_bw[letter][2]:
                        val = numbers_c[letter][2]
                    if val:
                        temp_chunk = m.m.goals["g"].pop()
                        m.m.goals["g"].add(actr.makechunk(typename=temp_chunk.typename, state=temp_chunk.state, arg1=temp_chunk.arg1, arg2=val, sum=str(temp_chunk.sum), end=temp_chunk.end, value=str(temp_chunk.value)))
                        #print(m.m.goals["g"])
                        #input()

            if re.search("RULE FIRED: retrieve_addition", sim.current_event.action):
                letter = str(list(m.m.goals["g"])[0].value)
                #print(letter)
                #print(m.m.visbuffers["visual_location"].recent)
                for i in all_pos[letter][y]:
                    m.m.visbuffers["visual_location"].recent.append(environ.stimuli[0][i])
                #print(m.m.visbuffers["visual_location"].recent)
                y = 0
                #input()
            if re.search("KEY PRESSED:", sim.current_event.action):
                #print(m.productions)
                times.append(sim.show_time() - init_time)
                break
    #input()

#5.17 for counting quantifiers

print(len(times))
print(sum(times)/len(times))

#7.4 for proportional quantifiers, but only 21 recorded (some problems with recording)
