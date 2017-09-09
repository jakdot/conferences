"""
A model of lexical decision.

You need to install pyactr to run it.

The model simulates lexicall access for the full list of Murray et al. items, coming from 16 frequency bands.
"""

import random
import pyactr as actr

environment = actr.Environment(focus_position=(320, 180))
model = actr.ACTRModel(environment=environment,\
                       subsymbolic=True,\
                       automatic_visual_search=False,\
                       activation_trace=False,\
                       retrieval_threshold=-10,\
                       latency_factor=0.13,\
                       latency_exponent=0.13,\
                       decay=0.5,\
                       motor_prepared=True,
                       eye_mvt_scaling_parameter=0.22,\
                       emma_noise=False)

actr.chunktype("goal", "state")
actr.chunktype("word", "form")

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR

# on average, 15 years of exposure is 112.5 million words

FOUND = list(reversed([542, 555, 566, 562, 570, 569, 577, 587, 592, 605, 603, 575, 620, 607, 622, 674]))

FREQ = {}
FREQ['guy'] = 242*112.5
FREQ['somebody'] = 92*112.5
FREQ['extend'] = 58*112.5
FREQ['dance'] = 40.5*112.5
FREQ['shape'] = 30.6*112.5
FREQ['besides'] = 23.4*112.5
FREQ['fit'] = 19*112.5
FREQ['dedicate'] = 16*112.5
FREQ['robot'] = 13.4*112.5
FREQ['tile'] = 11.5*112.5
FREQ['between'] = 10*112.5
FREQ['precedent'] = 9*112.5
FREQ['wrestle'] = 7*112.5
FREQ['resonate'] = 5*112.5
FREQ['seated'] = 3*112.5
FREQ['habitually'] = 1*112.5

model.productionstring(name="encoding word", string="""
    =g>
    isa     goal
    state   'start'
    =visual>
    isa     _visual
    value   =val
    ==>
    =g>
    isa     goal
    state   retrieving
    +g2>
    isa     word
    form    =val""")

model.productionstring(name="retrieving", string="""
    =g>
    isa     goal
    state   retrieving
    =g2>
    isa     word
    form    =val
    ==>
    =g>
    isa     goal
    state   retrieval_done
    +retrieval>
    isa     word
    form    =val""")

model.productionstring(name="can_recall", string="""
    =g>
    isa     goal
    state   retrieval_done
    ?retrieval>
    buffer  full
    state   free
    ==>
    ~g>
    +manual>
    isa     _manual
    cmd     press_key
    key     'J'""")

model.productionstring(name="cannot_recall", string="""
    =g>
    isa     goal
    state   retrieval_done
    ?retrieval>
    state   error
    ==>
    ~g>
    +manual>
    isa     _manual
    cmd     press_key
    key     'F'""")

if __name__ == "__main__":
    NITER = 10
    activation_dict = {key: [] for key in FREQ}
    time_dict = {key: [] for key in FREQ}
    for lemma in FREQ:
        print(lemma)
        for _ in range(NITER):
            model.decmems = {}
            model.set_decmem(data={})

            model.goals = {}

            dm = model.decmem
            for _ in range(int(FREQ[lemma])):
                dm.add(actr.makechunk(typename="word", form=lemma), time=random.randint(-SEC_IN_TIME, 0))
            word = {1: {'text': lemma, 'position': (320, 180)}}
            retrieval = model.retrieval

            model.set_goal("g")
            model.goal.add(actr.makechunk(nameofchunk='start', typename="goal", state='start'))

            model.set_goal("g2")
            model.goals["g2"].add(actr.makechunk(nameofchunk='start', typename="word", form=None))
            model.goals["g2"] = 0.2

            environment.current_focus = [320,180]

            model.model_parameters['motor_prepared'] = True

            sim = model.simulation(realtime=False, gui=True, trace=False, environment_process=environment.environment_process, stimuli=word, triggers='', times=2)
            while True:
                sim.step()
                if sim.current_event.action == "START RETRIEVAL":
                    activation_dict[lemma].append(retrieval.activation)
                if sim.current_event.action == "KEY PRESSED: J":
                    time_dict[lemma].append(sim.show_time())
                    break
                if sim.current_event.action == "KEY PRESSED: F":
                    break
    for i, key in enumerate(sorted(list(activation_dict.keys()), key=lambda x:FREQ[x])):
        #print(key, sum(activation_dict[key])/len(activation_dict[key]))
        print("word", "FREQ", "PREDICTED RT", "FOUND RT")
        print(key, FREQ[key]/112.5, sum(time_dict[key])/len(time_dict[key]), FOUND[i])
