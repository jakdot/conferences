"""
A model of lexical decision.

You need to install pyactr to run it.

The model simulates one stimulus-reaction process.
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
    model.decmems = {}
    model.set_decmem(data={})

    model.goals = {}

    dm = model.decmem
    dm.add(actr.makechunk(typename="word", form='habitually'), time=0)
    word = {1: {'text': 'habitually', 'position': (320, 180)}}
    retrieval = model.retrieval

    model.set_goal("g")
    model.goal.add(actr.makechunk(nameofchunk='start', typename="goal", state='start'))

    model.set_goal("g2")
    model.goals["g2"].add(actr.makechunk(nameofchunk='start', typename="word", form=None))
    model.goals["g2"] = 0.2

    environment.current_focus = [320,180]

    model.model_parameters['motor_prepared'] = True

    sim = model.simulation(realtime=True, gui=True, trace=True, environment_process=environment.environment_process, stimuli=word, triggers='', times=2)
    sim.run(2)
