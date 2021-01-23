# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import numpy as np
import matplotlib.pyplot as plt


# %%
v_idx = 0
u_idx = 1


# %%
def backward_euler(func, dfunc, past_states, constants, error_min = 1e-9, clip_lim = 20):
    temp_states = past_states.copy()
    error = np.ones_like(temp_states)
    max_error = 1
    while  max_error > error_min:
         
        f = func(temp_states, past_states, constants)
        df = dfunc(temp_states, past_states, constants)

        error = np.dot(np.linalg.inv(df), f)
        error = np.clip(error, -clip_lim, clip_lim)

        temp_states = np.subtract(temp_states, error)
    
        max_error = np.amax(np.absolute(error))
        print(max_error)
    return temp_states


# %%
def izhi_f(states, past_states, constants):
    F = np.zeros((2,1))
    
    F[0] = past_states[v_idx] -states[v_idx] + constants['del_t']*(0.04 * states[v_idx]**2 + 5*states[v_idx] + 140 - states[u_idx] + constants['i'])
    F[1] = past_states[u_idx] - states[u_idx] + constants['del_t']*constants['a']*(constants['b'] * states[v_idx] - states[u_idx])
    
    return F


# %%
def izhi_df(states, past_states, constants):
    
    j_f = np.zeros((2,2))
    
    j_f[0][0] = - 1 + constants['del_t']*(0.08*states[v_idx] + 5)
    j_f[0][1] = -constants['del_t']
    
    j_f[1][0] = constants['del_t']*constants['a']*constants['b']
    j_f[1][1] = - 1 - constants['del_t']*constants['a']
    
    return j_f


# %%
def compute(input_i, init_states, constants):
    ret_val = []
    past_states = init_states.copy()
    for i in range(input_i.shape[0]):
        constants['i'] = input_i[i]
        past_states = backward_euler(izhi_f, izhi_df, past_states, constants)
        if past_states[v_idx] > 30:
            past_states[v_idx] = constants['c']
            past_states[u_idx] += constants['d']
        ret_val.append(past_states)
    return np.array(ret_val).reshape(-1, past_states.shape[0])   

# %% [markdown]
# # 1

# %%
constants = {}
constants['a'] = 0.1
constants['b'] = 0.05
constants['c'] = -50
constants['d'] = 8

sample_rate = 50
max_t = 500
constants['del_t'] = 1/sample_rate

t = np.arange(0, max_t, step = constants['del_t'])
i_s = [20]

results = []
for i in i_s:
    input_i = np.full(t.shape[0], i)
    init_states = np.array([[-80],[0]])
    result = compute(input_i, init_states, constants)
    results.append(result)


# %%
plt.plot(t,results[-1])
plt.show()
