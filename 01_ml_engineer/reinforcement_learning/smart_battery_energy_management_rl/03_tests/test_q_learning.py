import numpy as np
from battery_rl.agents.q_learning import StateDiscretizer, TabularQLearningAgent

def test_discretizer_output_shape_and_range():
    d=StateDiscretizer(8,8,6,6,6)
    s=d.transform(np.array([0.5,0.2,0.8,0.1,0.9]))
    assert len(s)==5
    assert 0 <= s[0] < 8

def test_q_update_changes_value():
    d=StateDiscretizer(); a=TabularQLearningAgent(d,epsilon=0.0)
    obs=np.array([.5,.5,.5,.5,.5]); nxt=np.array([.6,.5,.5,.5,.6])
    before=a._values(d.transform(obs))[1]
    a.update(obs,1,1.0,nxt,False,[0,1,2])
    after=a._values(d.transform(obs))[1]
    assert after > before
