import random as rd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from brian2 import *
from eNose_data_loader import eNose_data_loader
import os
import pandas as pd
from pandas import DataFrame as Df
from sklearn.decomposition import PCA
from distinctipy import distinctipy
import copy
import pickle

'''
Unresolved problems:
1. 之後每個連結可能都需要獨立設置變數，如此才能有adaptive或max的g_syn值，目前是共用
2. 尚無加入NMDA

'''
############ LIF neuron model ################
class neuron():
    def __init__(self, neuron_model='LIF'):
        '''
        :param neuron_model: There are two options - 'LIF', 'hSTDP'
        There are several basic attribute in the model which is modfiable.
        For example, V_rest is the resting membrane potential. gleak is the leaking conductance.
        x_trace is the record for STDP.
        I_inject is the external inject current.
        E_AMPA is the AMPA poetential.
        '''
        if neuron_model == 'LIF':
            self.neuron_model = 'LIF'
            self.V_rest = -70. * mV  # resting potential
            self.v = self.V_rest
            self.x_trace = 0.
            self.taux = 15. * ms  # spike trace time constant
            self.gleak = 30. * nS  # leak conductance
            self.C = 300. * pF  # membrane capacitance
            self.x_trace = 0.
            self.I_inject = 0 * amp
            self.E_AMPA = 0. * mV
            self.E_GABA = -80 * mV
            self.E_Ach = 0 * mV
            self.tau_AMPA = 2 * ms
            self.tau_GABA = 5 * ms
            self.tau_Ach = 20 * ms
            self.g_AMPA = 0. * nS
            self.g_GABA = 0 * nS
            self.g_Ach = 0 * nS
            # ampa_max_cond = 50 * nS  # Ampa maximal conductance
            self.eqs = '''
                    dv/dt = (gleak*(V_rest-v)+I_inject+I_syn)/C: volt    (unless refractory)                     # voltage
                    dx_trace/dt = -x_trace/taux :1                          # spike trace
                    dg_AMPA/dt = -g_AMPA/tau_AMPA:siemens
                    dg_Ach/dt = -g_Ach/tau_Ach:siemens
                    dg_GABA/dt = -g_GABA/tau_GABA:siemens
                    I_AMPA = g_AMPA*(E_AMPA-v): amp
                    I_Ach = g_Ach*(E_Ach-v): amp
                    I_GABA = g_GABA*(E_GABA-v): amp
                    I_syn = I_AMPA + I_Ach + I_GABA :amp
                    tau_AMPA:second
                    E_AMPA : volt
                    tau_Ach:second
                    E_Ach : volt
                    tau_GABA:second
                    E_GABA : volt
                    rates : Hz                                              # input rates
                    selected_index : integer (shared)                       # active neuron
                    taux : second
                    C : farad
                    V_rest : volt
                    gleak : siemens
                    I_inject: amp        
                '''
        elif neuron_model == 'hSTDP':
            self.neuron_model = 'hSTDP'
            self.gleak = 30. * nS  # leak conductance
            self.C = 300. * pF  # membrane capacitance
            self.V_rest = -70 * mV
            self.v_lowpass1 = self.V_rest
            self.v_lowpass2 = self.V_rest
            self.v = self.V_rest
            self.v_homeo = 0 * mV
            self.I_inject = 0 * amp
            self.x_trace = 0
            self.Theta_low = self.V_rest  # depolarization threshold for plasticity
            self.x_reset = 1.  # spike trace reset value
            self.tau_lowpass1 = 40 * ms  # timeconstant for low-pass filtered voltage
            self.tau_lowpass2 = 30 * ms  # timeconstant for low-pass filtered voltage
            self.tau_homeo = 900 * ms  # homeostatic timeconstant
            self.taux = 15. * ms
            self.v_target = 12 * mV  # target depolarisation
            self.E_AMPA = 0. * mV
            self.E_GABA = -80 * mV
            self.E_Ach = 0 * mV
            self.tau_AMPA = 2 * ms
            self.tau_GABA = 5 * ms
            self.tau_Ach = 20 * ms
            self.g_AMPA = 0. * nS
            self.g_GABA = 0 * nS
            self.g_Ach = 0 * nS
            self.eqs = '''
                    dv/dt = (gleak*(V_rest-v) + I_inject + I_syn)/C: volt      # voltage
                    dv_lowpass1/dt = (v-v_lowpass1)/tau_lowpass1 : volt     # low-pass filter of the voltage
                    dv_lowpass2/dt = (v-v_lowpass2)/tau_lowpass2 : volt     # low-pass filter of the voltage
                    dv_homeo/dt = (v-V_rest-v_homeo)/tau_homeo : volt       # low-pass filter of the voltage
                    dg_AMPA/dt = -g_AMPA/tau_AMPA:siemens
                    dg_Ach/dt = -g_Ach/tau_Ach:siemens
                    dg_GABA/dt = -g_GABA/tau_GABA:siemens
                    dx_trace/dt = -x_trace/taux :1                          # spike trace
                    I_AMPA = g_AMPA*(E_AMPA-v): amp
                    I_Ach = g_Ach*(E_Ach-v): amp
                    I_GABA = g_GABA*(E_GABA-v): amp
                    I_syn = I_AMPA + I_Ach + I_GABA :amp
                    tau_AMPA:second
                    E_AMPA : volt
                    tau_Ach:second
                    E_Ach : volt
                    tau_GABA:second
                    E_GABA : volt
                    gleak : siemens
                    C : farad
                    V_rest : volt
                    I_inject : amp                                             # external current
                    tau_lowpass1 :second
                    tau_lowpass2 :second
                    tau_homeo : second
                    v_target: volt
                    x_reset : 1            # spike trace reset value
                    Theta_low: volt
                    taux:second
                    '''
    def print_attributes(self):
        for attribute, value in self.__dict__.items():
            print(f"{attribute}: {value}")

class connection():
    def __init__(self, connection_model='general', receptor='AMPA', w_max=200):
        if connection_model == 'general':
            s = general_synapses(receptor = receptor, model = 'simple', w_max = 200)
        elif connection_model == 'STDP':
            s = STDP_synapses(receptor=receptor,model='simple',w_max=w_max)
        for attribute, value in s.__dict__.items():
            setattr(self, attribute, value)

    def print_attributes(self):
        for attribute, value in self.__dict__.items():
            print(f"{attribute}: {value}")

class LIF_neuron():
    V_rest = -70. * mV  # resting potential
    v = V_rest
    x_trace = 0.
    taux = 15. * ms  # spike trace time constant
    gleak = 30. * nS  # leak conductance
    C = 300. * pF  # membrane capacitance
    x_trace = 0.
    I_inject = 0 * amp
    # I_syn = 0 * amp
    E_AMPA = 0.*mV
    E_GABA = -80 *mV
    E_Ach = 0*mV
    tau_AMPA = 2*ms
    tau_GABA = 5*ms
    tau_Ach = 20*ms
    g_AMPA = 0. * nS
    g_GABA = 0 * nS
    g_Ach = 0 * nS
    # ampa_max_cond = 50 * nS  # Ampa maximal conductance
    eqs = '''
        dv/dt = (gleak*(V_rest-v)+I_inject+I_syn)/C: volt    (unless refractory)                     # voltage
        dx_trace/dt = -x_trace/taux :1                          # spike trace
        dg_AMPA/dt = -g_AMPA/tau_AMPA:siemens
        dg_Ach/dt = -g_Ach/tau_Ach:siemens
        dg_GABA/dt = -g_GABA/tau_GABA:siemens
        I_AMPA = g_AMPA*(E_AMPA-v): amp
        I_Ach = g_Ach*(E_Ach-v): amp
        I_GABA = g_GABA*(E_GABA-v): amp
        I_syn = I_AMPA + I_Ach + I_GABA :amp
        tau_AMPA:second
        E_AMPA : volt
        tau_Ach:second
        E_Ach : volt
        tau_GABA:second
        E_GABA : volt
        rates : Hz                                              # input rates
        selected_index : integer (shared)                       # active neuron
        taux : second
        C : farad
        V_rest : volt
        gleak : siemens
        I_inject: amp        
    '''

class hSTDP_neuron():
    gleak = 30. * nS  # leak conductance
    C = 300. * pF  # membrane capacitance
    V_rest = -70 * mV
    v_lowpass1 = V_rest
    v_lowpass2 = V_rest
    v = V_rest
    v_homeo = 0 * mV
    I_inject = 0 * amp
    x_trace = 0
    Theta_low = V_rest  # depolarization threshold for plasticity
    x_reset = 1.  # spike trace reset value
    tau_lowpass1 = 40 * ms  # timeconstant for low-pass filtered voltage
    tau_lowpass2 = 30 * ms  # timeconstant for low-pass filtered voltage
    tau_homeo = 900 * ms  # homeostatic timeconstant
    taux = 15. * ms
    v_target = 12 * mV  # target depolarisation
    E_AMPA = 0. * mV
    E_GABA = -80 * mV
    E_Ach = 0 * mV
    tau_AMPA = 2 * ms
    tau_GABA = 5 * ms
    tau_Ach = 20 * ms
    g_AMPA = 0. * nS
    g_GABA = 0 * nS
    g_Ach = 0 * nS
    eqs = '''
        dv/dt = (gleak*(V_rest-v) + I_inject + I_syn)/C: volt      # voltage
        dv_lowpass1/dt = (v-v_lowpass1)/tau_lowpass1 : volt     # low-pass filter of the voltage
        dv_lowpass2/dt = (v-v_lowpass2)/tau_lowpass2 : volt     # low-pass filter of the voltage
        dv_homeo/dt = (v-V_rest-v_homeo)/tau_homeo : volt       # low-pass filter of the voltage
        dg_AMPA/dt = -g_AMPA/tau_AMPA:siemens
        dg_Ach/dt = -g_Ach/tau_Ach:siemens
        dg_GABA/dt = -g_GABA/tau_GABA:siemens
        dx_trace/dt = -x_trace/taux :1                          # spike trace
        I_AMPA = g_AMPA*(E_AMPA-v): amp
        I_Ach = g_Ach*(E_Ach-v): amp
        I_GABA = g_GABA*(E_GABA-v): amp
        I_syn = I_AMPA + I_Ach + I_GABA :amp
        tau_AMPA:second
        E_AMPA : volt
        tau_Ach:second
        E_Ach : volt
        tau_GABA:second
        E_GABA : volt
        gleak : siemens
        C : farad
        V_rest : volt
        I_inject : amp                                             # external current
        tau_lowpass1 :second
        tau_lowpass2 :second
        tau_homeo : second
        v_target: volt
        x_reset : 1            # spike trace reset value
        Theta_low: volt
        taux:second
        '''




class general_synapses():
    __Receptor_dict = {'AMPA':{'tau_syn':2*ms,'E_syn':0*mV},
                       'GABA':{'tau_syn':5*ms,'E_syn':-80*mV},
                       'Ach':{'tau_syn':20*ms,'E_syn':0*mV}}
    def __init__(self,receptor='AMPA',model='simple',w_max=200):
        if model == 'simple':
            self.syn_model = '''
            w : siemens
            w_max:siemens
            '''
            self.pre_eqs = f'''
            w = clip(w, 0*nS, w_max)
            g_{receptor}_post += w
            '''
            self.post_eqs = ""
            self.w_max = w_max * nS

class STDP_synapses_modified():
    '''
    要避免STDP動到0的部分:
    1. 每一個synapse個別去設w_max，原本是0的就設為w_max=0，如此在clip的時候就會把他clip在0 → 此步驟可以少一次運算
    2. 增設一個數值判定是否需要做STDP，最終的w就是乘上這個值
    '''
    def __init__(self,receptor='AMPA',model='simple',w_max=200):
        if model == 'simple':
            self.syn_model = f'''
                            w:siemens                # synaptic weight (ampa synapse)
                            dApre/dt = -Apre/taupre: 1 (clock-driven)
                            dApost/dt = -Apost/taupost: 1 (clock-driven)
                            w_max: siemens
                            tmpApre:1
                            tmpApost:1
                            taupre: second
                            taupost: second
                            '''

            # equations executed only when a presynaptic spike occurs
            self.pre_eqs = f'''
                            g_{receptor}_post += w                                                               # increment synaptic conductance
                            Apre += tmpApre
                            w = clip(w + Apost*nS, 0*nS, w_max)                                                           # hard bounds
                            '''

            # equations executed only when a postsynaptic spike occurs
            self.post_eqs = '''
                            Apost += tmpApost
                            w = clip(w + Apre*nS, 0*nS, w_max)                                                                     # hard bounds
                            '''
            self.tmpApre = 0.05
            self.tmpApost = -self.tmpApre
            self.taupre = 20 * ms
            self.taupost = 20 * ms
            self.w_max=w_max * nS


class STDP_synapses():
    '''
    要避免STDP動到0的部分:
    1. 每一個synapse個別去設w_max，原本是0的就設為w_max=0，如此在clip的時候就會把他clip在0 → 此步驟可以少一次運算
    2. 增設一個數值判定是否需要做STDP，最終的w就是乘上這個值
    '''
    def __init__(self,receptor='AMPA',model='simple',w_max=200):
        if model == 'simple':
            self.syn_model = f'''
                            w:siemens                # synaptic weight (ampa synapse)
                            dApre/dt = -Apre/taupre: 1 (clock-driven)
                            dApost/dt = -Apost/taupost: 1 (clock-driven)
                            w_max:siemens
                            tmpApre:1
                            tmpApost:1
                            taupre: second
                            taupost: second
                            '''

            # equations executed only when a presynaptic spike occurs
            self.pre_eqs = f'''
                            g_{receptor}_post += w                                                               # increment synaptic conductance
                            Apre += tmpApre
                            w = clip(w + Apost*nS, 0*nS, w_max)                                                           # hard bounds
                            '''

            # equations executed only when a postsynaptic spike occurs
            self.post_eqs = '''
                            Apost += tmpApost
                            w = clip(w + Apre*nS, 0*nS, w_max)                                                                     # hard bounds
                            '''
            self.tmpApre = 0.05
            self.tmpApost = -self.tmpApre
            self.taupre = 20 * ms
            self.taupost = 20 * ms
            self.w_max=w_max * nS

    # def describe_model(self):

class hSTDP_synapses_pre():
    def __init__(self,receptor='AMPA',model='simple',w_max=200):
        if model == 'simple':
            self.syn_model = f'''
                            w:siemens                # synaptic weight (ampa synapse)
                            max_cond: 1
                            A_LTD_u:1
                            w_minus:siemens
                            w_plus:siemens
                            w_max:siemens
                            A_LTP: 1
                            A_LTD:1
                        '''

            # equations executed only when a presynaptic spike occurs
            self.pre_eqs = f'''
                            g_{receptor}_post += w*max_cond                                                               # increment synaptic conductance
                            A_LTD_u = A_LTD*(v_homeo_pre**2/mV/v_target_pre)                                                             # metaplasticity
                            w_minus = A_LTD_u*(v_lowpass1_pre/mV - Theta_low_pre/mV)*int(v_lowpass1_pre/mV - Theta_low_pre/mV > 0)*nS  # synaptic depression
                            v_lowpass1_pre += 10*mV                                                                                        # mimics the depolarisation effect due to a spike
                            v_lowpass2_pre += 10*mV                                                                                        # mimics the depolarisation effect due to a spike
                            v_homeo_pre += 0.1*mV                                                                                          # mimics the depolarisation effect due to a spike
                            w = clip(w-w_minus, 0*nS, w_max)                                                           # hard bounds
                            '''

            # equations executed only when a postsynaptic spike occurs
            self.post_eqs = '''
                            w_plus = A_LTP*x_trace_pre*(v_lowpass2_pre/mV - Theta_low_pre/mV)*int(v_lowpass2_pre/mV - Theta_low_pre/mV > 0)*nS  # synaptic potentiation
                            w = clip(w+w_plus, 0*nS, w_max)                                                                     # hard bounds
                            '''
            self.max_cond = 1
            self.A_LTD = 1.5e-4  # depression amplitude
            self.A_LTP = 1.5e-2  # potentiation amplitude
            self.w_max=w_max * nS


class hSTDP_synapses_post():
    def __init__(self,receptor='AMPA',model='simple',w_max=200):
        if model == 'simple':
            self.syn_model = f'''
                            w:siemens                # synaptic weight (ampa synapse)
                            max_cond: 1
                            A_LTD_u:1
                            w_minus:siemens
                            w_plus:siemens
                            w_max:siemens
                            A_LTP: 1
                            A_LTD:1
                        '''

            # equations executed only when a presynaptic spike occurs
            self.pre_eqs = f'''
                            g_{receptor}_post += w*max_cond                                                               # increment synaptic conductance
                            A_LTD_u = A_LTD*(v_homeo_post**2/mV/v_target_post)                                                             # metaplasticity
                            w_minus = A_LTD_u*(v_lowpass1_post/mV - Theta_low_post/mV)*int(v_lowpass1_post/mV - Theta_low_post/mV > 0)*nS  # synaptic depression
                            w = clip(w-w_minus, 0*nS, w_max)                                                           # hard bounds
                            '''

            # equations executed only when a postsynaptic spike occurs
            self.post_eqs = '''
                            v_lowpass1_post += 10*mV                                                                                        # mimics the depolarisation effect due to a spike
                            v_lowpass2_post += 10*mV                                                                                        # mimics the depolarisation effect due to a spike
                            v_homeo_post += 0.1*mV                                                                                          # mimics the depolarisation effect due to a spike
                            w_plus = A_LTP*x_trace_pre*(v_lowpass2_post/mV - Theta_low_post/mV)*int(v_lowpass2_post/mV - Theta_low_post/mV > 0)*nS  # synaptic potentiation
                            w = clip(w+w_plus, 0*nS, w_max)                                                                     # hard bounds
                            '''
            self.max_cond = 1
            self.A_LTD = 1.5e-4  # depression amplitude
            self.A_LTP = 1.5e-2  # potentiation amplitude
            self.w_max=w_max * nS

class Stimulation_generator():
    def __init__(self):
        self.pca_bin_edges_dict = {}
        self.pca_bin_width_dict = {}
        self.pooled_data_dict = {}
        self.Odor_type = []
        self.Protocols = {}
        self.timeline = []
        self.event_number =0
        self.Neurons = {}
        self.pooled_max_dict={}

    def generate_spike(self, r, duration, dt=0.001):
        x = np.random.rand(duration)
        y = []
        for i in range(x.shape[0]):
            if x[i] <= r * dt:
                y.append(1)
            else:
                y.append(0)
        t = [i for i in range(len(y)) if y[i] == 1]
        return t

    def add_poisson_input(self, neuron_instance, neuron_number, strength, weight):
        external_poisson_input = PoissonInput(
            target=neurons, target_var="v", N=C_ext, rate=nu_ext, weight=J
        )

    def add_current(self, start_t, neuron_group, I, unit=amp, protocol_name=''):
        '''

        :param start_t:
        :param neuron_group:
        :param I: shape = (duration, neuron number)
        :param unit: amp
        :param protocol_name:
        :return:
        '''
        if protocol_name == '':
            protocol_name = f"custom {self.event_number}"
            self.event_number += 1
        end_t = start_t + len(I)
        self.timeline.append([protocol_name, start_t, end_t, 'custom','I_inject', neuron_group])
        self.Protocols[protocol_name] = np.array(I)*unit
        return

    def add_constant_current(self, start_t, end_t, neuron_group, current_mean, current_std=0,
                             notfixed=True, unit=amp, all=True, neuronids=[], stimulated_p=1, protocol_name=''):
        ## 'PN'
        #print("neuron_group = ", neuron_group)
        #print("self.Neurons = ", self.Neurons)
        neuron_number = self.Neurons[neuron_group].N
        if not protocol_name:
            protocol_name = f"Constant {self.event_number}"
            self.event_number += 1
        if neuronids:
            all = False
        duration = end_t - start_t
        current_list = []
        if current_std == 0:
            notfixed=False
        for i in range(neuron_number):
            if not all and i not in neuronids:
                continue
            if notfixed:
                variation = np.random.normal(0, current_std, duration)
            else:
                variation = np.random.normal(0, current_std)
            if rd.random()<stimulated_p:
                tmp = np.zeros(duration) + current_mean + variation
                current_list.append(tmp)
            else:
                tmp = np.zeros(duration) + variation
                current_list.append(tmp)
        self.timeline.append([protocol_name, start_t, end_t, 'constant','I_inject', neuron_group])
        self.Protocols[protocol_name] = np.array(current_list).transpose()*unit

        return

    def add_ramp_current(self, start_t, end_t, neuron_group, current_start, current_end,
                         current_std=0, notfixed=True, unit=amp, all=True, neuronids=[], stimulated_p=1,
                         protocol_name=''):
        neuron_number = self.Neurons[neuron_group].N
        if not protocol_name:
            protocol_name = f"Ramp {self.event_number}"
            self.event_number += 1
        if neuronids:
            all = False
        duration = end_t - start_t
        dIdt = (current_end - current_start)/duration
        current_list = []
        if current_std == 0:
            notfixed=False
        for i in range(neuron_number):
            if not all and i not in neuronids:
                continue
            if rd.random()<stimulated_p:
                if notfixed:
                    current_list.append([np.random.normal(current_start+dIdt*t,current_std) for t in range(duration)])
                else:
                    I = np.random.normal(current_start,current_std)
                    current_list.append([I+dIdt*t for t in range(duration)])
            else:
                if notfixed:
                    current_list.append([0+current_std for t in range(duration)])
                else:
                    current_list.append([0 for t in range(duration)])
        self.timeline.append([protocol_name, start_t, end_t, 'ramp','I_inject', neuron_group])
        self.Protocols[protocol_name] = np.array(current_list).transpose()*unit
        return

    def add_step_current(self, start_t, end_t, neuron_group, current_start,  current_end, level_num,
                         current_std=0, notfixed=True,
                         unit=amp, stimulated_p=1, all=True, neuronids=[], protocol_name=''):
        neuron_number = self.Neurons[neuron_group].N
        if not protocol_name:
            protocol_name = f"Step {self.event_number}"
            self.event_number += 1
        if neuronids:
            all = False
        duration = end_t - start_t
        dIdlevel = (current_end - current_start)/level_num
        period = np.floor(duration/level_num)
        current_list = []
        if current_std == 0:
            notfixed=False
        for i in range(neuron_number):
            if not all and i not in neuronids:
                continue
            if rd.random()<stimulated_p:
                if notfixed:
                    tmp = [np.random.normal(current_start+dIdlevel*level,current_std) for level in range(level_num) for t in range(period)]
                    tmp += [np.random.normal(current_start+dIdlevel*level,current_std) for _ in range(duration - len(tmp))]
                    current_list.append(tmp)
                else:
                    tmp = []
                    I = np.random.normal(current_start, current_std)
                    for level in range(level_num):
                        tmp += [I+dIdlevel*level for t in range(int(period))]
                    tmp += [np.random.normal(current_start + dIdlevel * level, current_std) for _ in
                            range(duration - len(tmp))]
                    current_list.append(tmp)
            else:
                if notfixed:
                    current_list.append([0+current_std for t in range(duration)])
                else:
                    current_list.append([0 for t in range(duration)])
        self.timeline.append([protocol_name, start_t, end_t, 'step','I_inject', neuron_group])
        self.Protocols[protocol_name] = np.array(current_list).transpose()*unit
        return

    def add_weight_normalization(self, time, connection_instance, target='output'):
        '''

        :param time:
        :param connection_instance:
        :param target:
        :return:
        '''
        protocol_name = f'{connection_instance}_{target}_{time}'
        self.timeline.append([protocol_name, time, time + 1, f'w_{target}_normalization', connection_instance])

    def pooling_odor_data(self):
        pooled_odor_dict = {}
        for odor in self.Odor_type:
            pooled_odor_dict[odor] = []
            for file in self.pooled_data_dict:
                if odor in file:
                    #print(len(self.pooled_data_dict[file].sensor_response_list))
                    pooled_odor_dict[odor].append(self.pooled_data_dict[file].sensor_response_list)
                    #for stimulation_index in range(len(self.pooled_data_dict[file].sensor_response_dict[0])):
                    #    stimulation_pooeld = []
                    #    for sensor_index in self.pooled_data_dict[file].sensor_response_dict:
                    #        stimulation_pooeld.append(self.pooled_data_dict[file].sensor_response_dict[sensor_index][stimulation_index])
                    #    pooled_odor_dict[odor].append(np.array(stimulation_pooeld))
        #print('pooled_odor_dict', pooled_odor_dict)
        self.pooled_odor_dict = pooled_odor_dict
        return

    def load_eNose(self):
        data_path = 'B1D1/'
        tmp_overlook_list = ['1090401dev1-Toluene-10ppm-ww.txt', '1090409dev1-NO2-5ppm-d.txt',
                             '1090409dev1-NO2-10ppm-d.txt']
        file_list = [i for i in os.listdir(data_path) if '.txt' in i and i not in tmp_overlook_list]
        file_list = file_list[:file_list.index("1090414dev1-Toluene-10ppm-w.txt")]
        tmp_overlook_list = [i for i in os.listdir(data_path) if i not in file_list]
        print("WARNING: Currently we overlooked the following data due to format inconsistency!\n", tmp_overlook_list)

        if not os.path.isfile(f"{eNose_data_loader.path}preprocessed_odor_data.pickle"):
            print("Initialization of odor data preprocessing!!!")
            pooled_data_dict = {}
            for file in file_list:
                data_loader = eNose_data_loader()
                data_loader.read_eNose_data_01(f"{data_path}{file}", plot=True)
                data_loader.divide_stimulation_into_subevents_2(plot=False)
                pooled_data_dict[file] = data_loader
            with open(f'{eNose_data_loader.path}preprocessed_odor_data.pickle', 'wb')as ff:
                pickle.dump(pooled_data_dict, ff)
        else:
            with open(f'{eNose_data_loader.path}preprocessed_odor_data.pickle', 'rb')as ff:
                pooled_data_dict = pickle.load(ff)
        # print("Odor data are loaded! We have the following odors:")
        Odor_type_list = [i.split("-")[1] for i in pooled_data_dict]
        Odor_type = list(dict.fromkeys(Odor_type_list))
        self.pooled_data_dict = pooled_data_dict
        self.Odor_type = Odor_type
        self.pooling_odor_data()
        # print(Odor_type)

    def odor_raw_voltage_stimulation(self, odor_type,duration = 100, current_std=0, scaling_factor=1.0 , notfixed=True,
                                     unit=amp, stimulation_index=0, absolute_value=True):
        if not self.pooled_data_dict:
            self.load_eNose()
        if current_std == 0:
            notfixed = False
        #duration = len(self.pooled_odor_dict[odor_type][0][0])
        self.odor_duration = duration
        current_list = []
        #for sensor_index in range(len(self.pooled_odor_dict[odor_type][stimulation_index])):
        for sensor_index in range(4200):
            if notfixed:
                variation = np.random.normal(0,current_std,duration)
            else:
                variation = np.random.normal(0,current_std)
            tmp = self.pooled_odor_dict[odor_type][stimulation_index][sensor_index]*scaling_factor + variation
            if absolute_value:
                tmp = np.abs(tmp)
            tmp_time = np.ones(self.odor_duration)
            tmp_time = tmp_time*tmp
            current_list.append(tmp_time)
        #print('current_list_length:',len(current_list))
        #print('current_list:',current_list)   #看看current_list的結構
        return np.array(current_list).transpose() * unit

    def add_odor_event(self, iteration_time=1, interval=30, odor_strength=10, absolute_value=False,
                        Answer_delay=3, Answer_strength=0.00000001,notfixed=False,unit=amp,
                        training_mode=False, event_name='',weight_norm=True):
        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        for iteration in range(iteration_time):
            if training_mode:
                rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    self.Protocols[protocol_name] = self.odor_raw_voltage_stimulation(odor, 0, odor_strength,
                                                                                      notfixed=notfixed,
                                                                                      unit=unit,
                                                                                      stimulation_index=stimulation_index,
                                                                                      absolute_value=absolute_value)
                    
                    #self.net[neuron_name].I_inject = self.Protocols[protocol_name][time][neuron_index]
                    print(protocol_name)
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")
                
                
                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_MBON'
                    self.timeline.append([protocol_name, Current_T+Answer_delay,
                                          Current_T+Answer_delay+self.odor_duration,
                                          odor, 'I_inject', 'MBON'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration,len(odor_choices)))
                        self.Protocols[protocol_name][:,self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                if not training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_n'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices))) - Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                                             target='input')
                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                                              target='input')
                #self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                #                              target='output')

                Current_T += self.odor_duration
        self.Current_T = Current_T

    def get_odor_stimulation_PCA(self, visualization=False, n_components=3, normalization=True):
        if not self.pooled_odor_dict:
            self.load_eNose()
        self.pca_dimension = n_components
        # print(self.pooled_odor_dict.keys())
        pooled_max_dict = {}
        for odor in self.pooled_odor_dict:
            pooled_data = []
            for trial in range(len(self.pooled_odor_dict[odor])):
                tmp_data = []
                for sensor_id in range(len(self.pooled_odor_dict[odor][trial])):
                    data = self.pooled_odor_dict[odor][trial][sensor_id]
                    tmp_data.append(data[np.argmax(np.abs(data))])
                pooled_data.append(tmp_data)
            pooled_max_dict[odor] = pooled_data
            '''
            odor can be 'SO2'
            pooled_data =
            [[sensor1_max sensor2_max....] -> trial 1
             [
            ]
            '''
        all_data = []
        index_collection = []
        for odor in pooled_max_dict:
            start = len(all_data)
            all_data += pooled_max_dict[odor]
            end = len(all_data)
            index_collection.append((start, end, odor))
        all_data = np.array(all_data)

        pca = PCA(n_components=n_components)
        pca.fit(all_data)
        all_data = pca.transform(all_data)

        if normalization:
            all_data = all_data + abs(np.min(all_data,axis=0))

        self.pca_data_pooled = all_data

        for i in range(len(index_collection)):
            start, end, odor = index_collection[i]
            pooled_max_dict[odor] = all_data[start:end]
        self.pooled_max_dict = pooled_max_dict

        '''
        data format is [[pca1 pca2 pca3],[pca1 pca2 pca3], ....]
        '''

        if visualization:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for label in index_collection:
                plt.plot(all_data[label[0]:label[1], 0], all_data[label[0]:label[1], 1], all_data[label[0]:label[1], 2],
                         '.', color=(rd.random(), rd.random(), rd.random()), label=label[2])
            plt.legend()
            plt.show()

    def add_odor_event_pca_digitize(self, iteration_time=1, duration=100, interval=30, odor_strength=10,
                               Answer_delay=3, Answer_strength=0.00000001, notfixed=False, unit=amp,
                               training_mode=False, event_name='', weight_norm=True, PN_num_scale=1):
        if not event_name:
            if training_mode:
                event_name = 'train'
            else:
                event_name = 'test'

        self.odor_duration = duration
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()

        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        for iteration in range(iteration_time):
            # if training_mode:
            #     rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    '''
                     odor_type, duration=20, notfixed=True,
                                  scaling_factor=1.0, unit=amp, stimulation_index=0,
                                  current_std=0.1
                    '''
                    self.Protocols[protocol_name] = self.odor_strength_stimulation_predivide(odor, duration=duration,
                                                                                             scaling_factor=odor_strength,
                                                                                             notfixed=notfixed,
                                                                                             unit=unit,
                                                                                             stimulation_index=stimulation_index,
                                                                                             current_std=0,
                                                                                             PN_num_scale=PN_num_scale,
                                                                                             )
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")

                # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_ALLN_p'
                # self.timeline.append([protocol_name, Current_T,
                #                       Current_T + self.odor_duration,
                #                       odor, 'I_inject', 'ALLN_p'])
                # if protocol_name not in self.Protocols:
                #     self.Protocols[protocol_name] = np.ones((self.odor_duration, 1), dtype=float) * 0.00000001 * amp
                #
                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_p'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_p'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices)))
                        self.Protocols[protocol_name][:, self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                # #
                if not training_mode:
                    # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    # self.timeline.append([protocol_name, Current_T + Answer_delay,
                    #                       Current_T + Answer_delay + self.odor_duration,
                    #                       odor, 'I_inject', 'Answer_n'])
                    #
                    # if protocol_name not in self.Protocols:
                    #     self.Protocols[protocol_name] = np.zeros(
                    #         (self.odor_duration, len(odor_choices))) - Answer_strength
                    #     self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                    #
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Forced'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Forced_answer'])
                    #
                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, 1)) + Answer_strength * 0.5
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                #                               target='input')
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                #                               target='output')
                # #
                # # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                # #                               target='output')

                Current_T += self.odor_duration
            # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
            #                           target='input')

            self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                                      target='output')

        self.Current_T = Current_T

    def odor_strength_stimulation_predivide(self, odor_type, duration=100, notfixed=True,
                                            scaling_factor=1.0, unit=amp, stimulation_index=0,
                                            current_std=0.1, PN_num_scale=1,padding=0):
        '''
        Note:
        20221214 implemeted. This function load pca transformed max response.
        In addition, we have made shift in pca function. Thus, the negative value has been shifted >=0.
        :param odor_type: Which odor, ex. SO2
        :param duration: Since the temporal dynamics has been ignored, it is the time for neuron to fire and train. (ms)
        :param notfixed: more noise or not
        :param scaling_factor: This factor adjust the current amplitude which is critical for firing.
        :param unit: amp.
        :param stimulation_index: Since every odor has a bunch of stimulation trial. the index specifies which trial.
        :param current_std: the noise of the current.
        :return:
        '''
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()
        if odor_type not in self.pooled_max_dict:
            raise BaseException("No such odor in the dataset")
        if current_std == 0:
            notfixed = False
        self.odor_duration = duration
        current_list = []
        if not self.pca_bin_edges_dict:
            pca_bin_edges_dict = {}
            pca_bin_width_dict = {}
            min_values = np.min(self.pca_data_pooled, axis=0)
            max_values = np.max(self.pca_data_pooled, axis=0)
            for pca_index in range(self.pca_dimension):
                group_num = PN_num_scale
                bin_width = (max_values[pca_index] - min_values[pca_index]) / (group_num - 1)
                pca_bin_width_dict[pca_index] = bin_width
                # Calculate the bin edges based on the minimum and maximum values in the data
                bin_edges = np.linspace(min_values[pca_index], max_values[pca_index], group_num)
                pca_bin_edges_dict[pca_index] = bin_edges
            self.pca_bin_edges_dict = pca_bin_edges_dict
            self.pca_bin_width_dict = pca_bin_width_dict
        else:
            pca_bin_edges_dict = self.pca_bin_edges_dict
            pca_bin_width_dict = self.pca_bin_width_dict

        # print(pca_bin_edges_dict)
        for pn_num in range(PN_num_scale):
            for pca_index in range(self.pca_dimension):
                bin_width = pca_bin_width_dict[pca_index] ## bin width
                bin_edges = pca_bin_edges_dict[pca_index]
                s = self.pooled_max_dict[odor_type][stimulation_index][pca_index] - 0.5 * bin_width # stimulus strength
                bin_index = np.digitize(s, pca_bin_edges_dict[pca_index], right=True) # bin index (shift 0.5 d)
                strength = (s - bin_edges[bin_index] + bin_width)/bin_width
                tmp = np.zeros(duration) + (2) * scaling_factor * (bin_index == pn_num)
                # tmp = np.zeros(duration) + (strength * 0.75 + 0.6) * scaling_factor * (bin_index == pn_num)
                current_list.append(tmp)
        return np.array(current_list).transpose() * unit

    def odor_strength_stimulation(self, odor_type, duration=100, notfixed=True,
                                  scaling_factor=1.0, unit=amp, stimulation_index=0,
                                  current_std=0.1, PN_num_scale=1):
        '''
        Note:
        20221214 implemeted. This function load pca transformed max response.
        In addition, we have made shift in pca function. Thus, the negative value has been shifted >=0.
        :param odor_type: Which odor, ex. SO2
        :param duration: Since the temporal dynamics has been ignored, it is the time for neuron to fire and train. (ms)
        :param notfixed: more noise or not
        :param scaling_factor: This factor adjust the current amplitude which is critical for firing.
        :param unit: amp.
        :param stimulation_index: Since every odor has a bunch of stimulation trial. the index specifies which trial.
        :param current_std: the noise of the current.
        :return:
        '''
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()
        if odor_type not in self.pooled_max_dict:
            raise BaseException("No such odor in the dataset")
        if current_std == 0:
            notfixed = False
        self.odor_duration = duration
        current_list = []
        for pn_num in range(PN_num_scale):
            for pca_index in range(self.pca_dimension):
                if notfixed:
                    variation = np.random.normal(0, current_std, duration)
                else:
                    variation = np.random.normal(0, current_std)
                tmp = np.array([self.pooled_max_dict[odor_type][stimulation_index][pca_index] * scaling_factor/
                                (pn_num+1) for _ in range(duration)]) + variation
                current_list.append(tmp)


        return np.array(current_list).transpose() * unit

    def odor_strength_stimulation_mix(self, odor_type, duration=100, notfixed=True,
                                  scaling_factor=1.0, unit=amp, stimulation_index=0,
                                  current_std=0.1, PN_num_scale=1, N_multi=0):
        '''
        Note:
        20221214 implemeted. This function load pca transformed max response.
        In addition, we have made shift in pca function. Thus, the negative value has been shifted >=0.
        :param odor_type: Which odor, ex. SO2
        :param duration: Since the temporal dynamics has been ignored, it is the time for neuron to fire and train. (ms)
        :param notfixed: more noise or not
        :param scaling_factor: This factor adjust the current amplitude which is critical for firing.
        :param unit: amp.
        :param stimulation_index: Since every odor has a bunch of stimulation trial. the index specifies which trial.
        :param current_std: the noise of the current.
        :return:
        '''
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()
        if odor_type not in self.pooled_max_dict:
            raise BaseException("No such odor in the dataset")
        if current_std == 0:
            notfixed = False
        self.odor_duration = duration
        current_list = []
        for pn_num in range(PN_num_scale):
            for pca_index in range(self.pca_dimension):
                if notfixed:
                    variation = np.random.normal(0, current_std, duration)
                else:
                    variation = np.random.normal(0, current_std)
                tmp = np.array([self.pooled_max_dict[odor_type][stimulation_index][pca_index] * scaling_factor/
                                (pn_num+1) for _ in range(duration)]) + variation
                current_list.append(tmp)
        if N_multi > 0 and self.multi_set == False:
            self.multi_set = True
            tmp_a = []
            for i_multi in range(N_multi):
                tmp = np.zeros(duration)
                a = np.random.random(self.pca_dimension)
                a[:np.random.randint(0, self.pca_dimension - 2)] = 0
                a = a / a.sum() * (np.random.random() * 0.0000001)
                np.random.shuffle(a)
                tmp_a.append(a)
            self.multi_composition = tmp_a
        for i_multi in range(N_multi):
            a = self.multi_composition[i_multi]
            for pca_index in range(self.pca_dimension):
                if notfixed:
                    variation = np.random.normal(0, current_std, duration)
                else:
                    variation = np.random.normal(0, current_std)
                tmp += (np.array([self.pooled_max_dict[odor_type][stimulation_index][pca_index] * scaling_factor/
                                (pn_num+1) for _ in range(duration)]) + variation) * a[pca_index]
            current_list.append(tmp)

        return np.array(current_list).transpose() * unit

    def add_odor_event_pca_mix(self, iteration_time=1, duration=100, interval=30, odor_strength=10,
                        Answer_delay=3, Answer_strength=0.00000001,notfixed=False,unit=amp,
                        training_mode=False, event_name='',weight_norm=True, PN_num_scale=1, N_multi=0):
        if not event_name:
            if training_mode:
                event_name = 'train'
            else:
                event_name = 'test'

        self.odor_duration = duration
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()

        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        for iteration in range(iteration_time):
            # if training_mode:
            #     rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    '''
                     odor_type, duration=20, notfixed=True,
                                  scaling_factor=1.0, unit=amp, stimulation_index=0,
                                  current_std=0.1
                    '''
                    self.Protocols[protocol_name] = self.odor_strength_stimulation_mix(odor, duration=duration,
                                                                                   scaling_factor=odor_strength,
                                                                                   notfixed=notfixed,
                                                                                   unit=unit,
                                                                                   stimulation_index=stimulation_index,
                                                                                   current_std=0,
                                                                                   PN_num_scale=PN_num_scale,
                                                                                   N_multi=N_multi
                                                                                   )
                    # print(protocol_name)
                    # print(self.Protocols[protocol_name].shape)
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")

                # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_ALLN_p'
                # self.timeline.append([protocol_name, Current_T,
                #                       Current_T + self.odor_duration,
                #                       odor, 'I_inject', 'ALLN_p'])
                # if protocol_name not in self.Protocols:
                #     self.Protocols[protocol_name] = np.ones((self.odor_duration, 1), dtype=float) * 0.00000001 * amp
                #
                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_p'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_p'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices)))
                        self.Protocols[protocol_name][:, self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                # #
                if not training_mode:
                    # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    # self.timeline.append([protocol_name, Current_T + Answer_delay,
                    #                       Current_T + Answer_delay + self.odor_duration,
                    #                       odor, 'I_inject', 'Answer_n'])
                #
                    # if protocol_name not in self.Protocols:
                    #     self.Protocols[protocol_name] = np.zeros(
                    #         (self.odor_duration, len(odor_choices))) - Answer_strength
                    #     self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Forced'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Forced_answer'])
                #
                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, 1)) + Answer_strength * 0.5
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                #                               target='input')
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                #                               target='output')
                # #
                # # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                # #                               target='output')

                Current_T += self.odor_duration
            # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
            #                           target='input')

            # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
            #                           target='output')

        self.Current_T = Current_T


    def add_odor_event_pca(self, iteration_time=1, duration=100, interval=30, odor_strength=10,
                        Answer_delay=3, Answer_strength=0.00000001,notfixed=False,unit=amp,
                        training_mode=False, event_name='',weight_norm=True, PN_num_scale=2):
        if not event_name:
            if training_mode:
                event_name = 'train'
            else:
                event_name = 'test'

        self.odor_duration = duration
        if not self.pooled_max_dict:
            if not self.pooled_odor_dict:
                self.load_eNose()
            self.get_odor_stimulation_PCA()

        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        print("odor_choices:",odor_choices)
        for iteration in range(iteration_time):
            # if training_mode:
            #     rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    '''
                     odor_type, duration=20, notfixed=True,
                                  scaling_factor=1.0, unit=amp, stimulation_index=0,
                                  current_std=0.1
                    '''
                    self.Protocols[protocol_name] = self.odor_strength_stimulation(odor, duration=duration,
                                                                                   scaling_factor=odor_strength,
                                                                                   notfixed=notfixed,
                                                                                   unit=unit,
                                                                                   stimulation_index=stimulation_index,
                                                                                   current_std=0,
                                                                                   PN_num_scale=PN_num_scale
                                                                                   )
                    print(protocol_name)
                    print(self.Protocols[protocol_name].shape)
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")

                # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_ALLN_p'
                # self.timeline.append([protocol_name, Current_T,
                #                       Current_T + self.odor_duration,
                #                       odor, 'I_inject', 'ALLN_p'])
                # if protocol_name not in self.Protocols:
                #     self.Protocols[protocol_name] = np.ones((self.odor_duration, 1), dtype=float) * 0.00000001 * amp
                #
                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_MBON'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'MBON'])
                    #protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_p'
                    #self.timeline.append([protocol_name, Current_T + Answer_delay,
                    #                      Current_T + Answer_delay + self.odor_duration,
                    #                      odor, 'I_inject', 'Answer_p'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices)))
                        self.Protocols[protocol_name][:, self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                # #
                if not training_mode:
                    # protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    # self.timeline.append([protocol_name, Current_T + Answer_delay,
                    #                       Current_T + Answer_delay + self.odor_duration,
                    #                       odor, 'I_inject', 'Answer_n'])
                #
                    # if protocol_name not in self.Protocols:
                    #     self.Protocols[protocol_name] = np.zeros(
                    #         (self.odor_duration, len(odor_choices))) - Answer_strength
                    #     self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Forced'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Forced_answer'])
                #
                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, 1)) + Answer_strength * 0.5
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                #                               target='input')
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                #                               target='output')
                # #
                # # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                # #                               target='output')

                Current_T += self.odor_duration
            self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                                      target='input')
            self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                                      target='output')

        self.Current_T = Current_T


    def add_odor_event_max(self, iteration_time=1, interval=30, odor_strength=10, absolute_value=False,
                        Answer_delay=3, Answer_strength=0.00000001,notfixed=False,unit=amp,
                        training_mode=False, event_name='',weight_norm=True):
        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        for iteration in range(iteration_time):
            if training_mode:
                rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    self.Protocols[protocol_name] = self.odor_raw_voltage_stimulation(odor, 0, odor_strength,
                                                                                      notfixed=notfixed,
                                                                                      unit=unit,
                                                                                      stimulation_index=stimulation_index,
                                                                                      absolute_value=absolute_value)
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")

                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_ALLN_p'
                self.timeline.append([protocol_name, Current_T,
                                      Current_T + self.odor_duration,
                                      odor, 'I_inject', 'ALLN_p'])
                if protocol_name not in self.Protocols:
                    self.Protocols[protocol_name] = np.ones((self.odor_duration, 1), dtype=float) * 0.00000001 * amp

                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_p'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_p'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices)))
                        self.Protocols[protocol_name][:, self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                if not training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_n'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, len(odor_choices))) - Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Forced'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Forced_answer'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, 1)) + Answer_strength * 0.5
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                                              target='input')
                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                                              target='output')
                #
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                #                               target='output')

                Current_T += self.odor_duration
        self.Current_T = Current_T

    def add_odor_event_v2(self, iteration_time=1, interval=30, odor_strength=10, absolute_value=False,
                        Answer_delay=3, Answer_strength=0.00000001,notfixed=False,unit=amp,
                        training_mode=False, event_name='',weight_norm=True):
        Current_T = self.Current_T
        print(f"Setting: {event_name}")
        odor_choices = copy.deepcopy(self.target_odor_list)
        for iteration in range(iteration_time):
            if training_mode:
                rd.shuffle(odor_choices)
            for odor in odor_choices:
                Current_T += interval
                if training_mode:
                    stimulation_index = rd.choice(self.odor_training_list_dict[odor])
                else:
                    stimulation_index = rd.choice(self.odor_test_list_dict[odor])
                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_PN'
                self.timeline.append([protocol_name, Current_T, Current_T + self.odor_duration,
                                      odor, 'I_inject', 'PN'])
                if protocol_name not in self.Protocols:
                    self.Protocols[protocol_name] = self.odor_raw_voltage_stimulation(odor, 0, odor_strength,
                                                                                      notfixed=notfixed,
                                                                                      unit=unit,
                                                                                      stimulation_index=stimulation_index,
                                                                                      absolute_value=absolute_value)
                else:
                    raise BaseException("Duplicate protocol occurs! Please rename your protocol_name!")

                protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_ALLN_p'
                self.timeline.append([protocol_name, Current_T,
                                          Current_T + self.odor_duration,
                                          odor, 'I_inject', 'ALLN_p'])
                if protocol_name not in self.Protocols:
                    self.Protocols[protocol_name] = np.ones((self.odor_duration, 1),dtype=float) * 0.00000001 *amp


                if training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_p'
                    self.timeline.append([protocol_name, Current_T+Answer_delay,
                                          Current_T+Answer_delay+self.odor_duration,
                                          odor, 'I_inject', 'Answer_p'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration,len(odor_choices)))
                        self.Protocols[protocol_name][:,self.target_odor_list.index(odor)] = Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp
                #
                if not training_mode:
                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Answer_n'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Answer_n'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros((self.odor_duration, len(odor_choices))) - Answer_strength
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                    protocol_name = f'{event_name}_{iteration}_{odor}_{stimulation_index}_Forced'
                    self.timeline.append([protocol_name, Current_T + Answer_delay,
                                          Current_T + Answer_delay + self.odor_duration,
                                          odor, 'I_inject', 'Forced_answer'])

                    if protocol_name not in self.Protocols:
                        self.Protocols[protocol_name] = np.zeros(
                            (self.odor_duration, 1)) + Answer_strength*0.5
                        self.Protocols[protocol_name] = self.Protocols[protocol_name] * amp

                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("PN", "KC"),
                                              target='input')
                self.add_weight_normalization(Current_T + self.odor_duration + 5, ("KC", "MBON"),
                                             target='output')
                #
                # self.add_weight_normalization(Current_T + self.odor_duration + 5, ("MBONinhibitor", "MBON"),
                #                               target='output')

                Current_T += self.odor_duration
        self.Current_T = Current_T

    def prepare_eNose_data(self, target_odor_list, training_ratio=0.8,seed=0):
        if seed: rd.seed(seed)
        odor_choices = copy.deepcopy(target_odor_list)
        odor_training_list_dict = {}
        odor_test_list_dict = {}
        ## Divide data into training and test
        for odor in odor_choices:
            odor_training_list_dict[odor] = rd.sample(
                [i for i in range(len(self.pooled_odor_dict[odor]))], int(training_ratio*len(self.pooled_odor_dict[odor])))
            odor_test_list_dict[odor] = [i for i in range(len(self.pooled_odor_dict[odor])) if i not in odor_training_list_dict[odor]]
        self.odor_training_list_dict = odor_training_list_dict
        self.odor_duration = 100
        #self.odor_duration = self.pooled_odor_dict[odor][0].shape[1]
        self.odor_choices = odor_choices
        self.Current_T = 0
        self.target_odor_list = target_odor_list
        self.odor_test_list_dict = odor_test_list_dict
        #print("odor_training_list_dict length:", len(odor_training_list_dict))
        #print("odor_training_list_dict:", odor_training_list_dict)
        #print("odor_test_list_dict length:", len(odor_test_list_dict))
        #print("odor_test_list_dict:", odor_test_list_dict)

    def plot_timeline(self,show=True, file_name=""):
        # stimulated_neurons = [f"{protocol[-3]}_{protocol[-2]}_{protocol[-1]}" for protocol in self.timeline]
        stimulated_neurons = [protocol[0] for protocol in self.timeline]
        neuron_color_dict = {}
        record = []
        for neuron in stimulated_neurons:
            if neuron not in neuron_color_dict:
                neuron_color_dict[neuron] = (rd.random(),rd.random(),rd.random())
        for protocol_index, protocol in enumerate(self.timeline):
            if f"{protocol[0]}" not in record:
                plt.plot([protocol[1],protocol[2]],[protocol_index+1,protocol_index+1],
                         color=neuron_color_dict[f"{protocol[0]}"],label=f"{protocol[0]}")
                record.append(f"{protocol[0]}")
            else:
                plt.plot([protocol[1], protocol[2]], [protocol_index + 1, protocol_index + 1],
                         color=neuron_color_dict[f"{protocol[0]}"])
            # if f"{protocol[-3]}_{protocol[-2]}_{protocol[-1]}" not in record:
            #     plt.plot([protocol[1],protocol[2]],[protocol_index+1,protocol_index+1],
            #              color=neuron_color_dict[f"{protocol[-3]}_{protocol[-2]}_{protocol[-1]}"],label=f"{protocol[-3]}_{protocol[-2]}_{protocol[-1]}")
            #     record.append(f"{protocol[-2]}_{protocol[-1]}")
            # else:
            #     plt.plot([protocol[1], protocol[2]], [protocol_index + 1, protocol_index + 1],
            #              color=neuron_color_dict[f"{protocol[-3]}_{protocol[-2]}_{protocol[-1]}"])
        plt.legend()
        plt.title("Timeline")
        if file_name:
            plt.savefig(file_name)
        if show:
            plt.show()
        plt.close()

class Configuration_generator():
    def __init__(self):
        self.Neurons = {}
        self.Connections = {}
        self.Neuron_model_list = ['LIF', 'hSTDP']
        self.Neuron_instance_dict = {}
        self.spike_generator_dict = {}
        self.Connection_copy = {}

    def batch_add_synapse_type_dict(self,data, neuron_type_transmitter_dict={}, weight_scaling_dict={}, synapse_model_dict={}, synapse_neurotransmitter_dict={}):
        '''
        :param data: connection data which is the standard format from eFlyplot
        :param neuron_type_transmitter_dict:  specify the neurotransmitter for a neuron.
                                            all synapses from this neuron will be set to the corresponding neurotransmitter type.
                                            (if there are two neurotransmitter, the user have to define in synapse_neurotransmitter_dict)
        :param weight_scaling_dict: specify the connection weight scaling for specific type connections
        :param synapse_model_dict: specify the eqs of the synapse
        :param synapse_neurotransmitter_dict: if there are some specific neurotransmitter for a synapse, the user can define it here.
        :return:
        '''
        synapse_type_dict = {}
        for connection in data:
            discard, up_id, up_instance, up_type, \
            down_id, down_instance, down_type, weight, roi = connection
            if up_type in neuron_type_transmitter_dict:
                synapse_type_dict[(up_type,down_type)] = ['general',neuron_type_transmitter_dict[up_type],1]
            if (up_type,down_type) in synapse_model_dict:
                if (up_type,down_type) not in synapse_type_dict:
                    synapse_type_dict[(up_type, down_type)] = [synapse_model_dict[(up_type,down_type)], 'AMPA', 1]
                else:
                    synapse_type_dict[(up_type, down_type)][0] = synapse_model_dict[(up_type, down_type)]
            if (up_type,down_type) in weight_scaling_dict:
                if (up_type, down_type) not in synapse_type_dict:
                    synapse_type_dict[(up_type, down_type)] = ['general', 'AMPA', weight_scaling_dict[(up_type,down_type)]]
                else:
                    synapse_type_dict[(up_type, down_type)][2] = weight_scaling_dict[(up_type,down_type)]
            if (up_type,down_type) in synapse_neurotransmitter_dict:
                synapse_type_dict[(up_type, down_type)] = synapse_neurotransmitter_dict[(up_type,down_type)]
        return synapse_type_dict

    def load_connections_from_connection_table(self, data=[], file_name='', upstream_neuron_list=[], downstream_neuron_list=[],
                                 weight_threshold=0, neuron_type_dict={}, synapse_type_dict={}, up_down_intersection=True):

        ## We can get neuron model from neuron dict and get neuron id in the matrix by neuron model dict
        self.upstream_of_neuron_dict = {}
        self.downstream_of_neuron_dict = {}
        if not data and file_name:
            data = pd.read_excel(file_name).values.tolist()
        Neuron_dict = {}
        Neuron_model_dict = {}
        for neuronal_model in self.Neuron_model_list:
            Neuron_model_dict[neuronal_model] = []
        Connection_dict = {}

        for connection in data:
            discard, up_id, up_instance, up_type,\
                down_id,down_instance,down_type,weight, roi = connection
            if weight < weight_threshold:
                continue
            if upstream_neuron_list or downstream_neuron_list:
                if not up_down_intersection:
                    if up_type not in upstream_neuron_list and down_type not in downstream_neuron_list:
                        continue
                else:
                    if up_type not in upstream_neuron_list or down_type not in downstream_neuron_list:
                        continue
            source = f'{up_instance}_{up_id}'.replace(">","_to_").replace("(","_").replace(")","_").replace("'","p").replace("/","_and_").replace("-","")
            target = f'{down_instance}_{down_id}'.replace(">","_to_").replace("(","_").replace(")","_").replace("'","p").replace("/","_and_").replace("-","")

            if source not in Neuron_dict:
                if up_type not in neuron_type_dict:
                    Neuron_dict[source] = 'LIF'
                    Neuron_model_dict['LIF'].append(source)
                elif neuron_type_dict[up_type] == 'hSTDP':
                    Neuron_dict[source] = 'hSTDP'
                    Neuron_model_dict['hSTDP'].append(source)
                else:
                    raise BaseException(f"Currently doesn't support {neuron_type_dict[up_type]} neuron model!")
                print(f"Add {source} into network")
            if target not in Neuron_dict:
                if down_type not in neuron_type_dict:
                    Neuron_dict[target] = 'LIF'
                    Neuron_model_dict['LIF'].append(target)
                elif neuron_type_dict[down_type] == 'hSTDP':
                    Neuron_dict[target] = 'hSTDP'
                    Neuron_model_dict['hSTDP'].append(target)
                else:
                    raise BaseException(f"Currently doesn't support {neuron_type_dict[down_type]} neuron model!")
                print(f"Add {target} into network")

            if (up_type,down_type) not in synapse_type_dict:
                r = 'AMPA'
                s = 'general'
            elif synapse_type_dict[(up_type,down_type)][0] == 'general':
                s, r, ws = synapse_type_dict[(up_type,down_type)]
                weight = ws * weight
            elif synapse_type_dict[(up_type,down_type)][0] == 'hSTDP_post':
                raise BaseException("Under development!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            Connection_dict[(source, target)] = [r, s, weight]
            if target not in self.upstream_of_neuron_dict:
                self.upstream_of_neuron_dict[target] = []
            self.upstream_of_neuron_dict[target].append(source)

            if source not in self.downstream_of_neuron_dict:
                self.downstream_of_neuron_dict[source] = []
            self.downstream_of_neuron_dict[source].append(target)
            print(f"Add {source} to {target} {r},{s}")

        Weight_dict = {}
        if Neuron_model_dict['LIF']:
            self.add_LIF_neuron('LIF', number=len(Neuron_model_dict['LIF']),eqs=LIF_neuron.eqs)
            Weight_dict[("LIF","LIF")] = np.zeros((len(Neuron_model_dict['LIF']), len(Neuron_model_dict['LIF'])))
        if Neuron_model_dict['hSTDP']:
            self.add_LIF_neuron('hSTDP', number=len(Neuron_model_dict['hSTDP']), eqs=hSTDP_neuron.eqs)
            Weight_dict[("hSTDP","hSTDP")] = np.zeros((len(Neuron_model_dict['hSTDP']), len(Neuron_model_dict['hSTDP'])))
        if Neuron_model_dict['LIF'] and Neuron_model_dict['hSTDP']:
            Weight_dict[("LIF","hSTDP")] = np.zeros((len(Neuron_model_dict['LIF']), len(Neuron_model_dict['hSTDP'])))
            Weight_dict[("hSTDP","LIF")] = np.zeros((len(Neuron_model_dict['hSTDP']), len(Neuron_model_dict['LIF'])))
        print("Set up Neuron models")

        for up_type in Neuron_model_dict:
            for down_type in Neuron_model_dict:
                for i, neuron_name_i in enumerate(Neuron_model_dict[up_type]):
                    for j, neuron_name_j in enumerate(Neuron_model_dict[down_type]):
                        if (neuron_name_i,neuron_name_j) in Connection_dict:
                            print(i,j, Connection_dict[(neuron_name_i,neuron_name_j)])
                            print(Weight_dict[(up_type,down_type)].shape)
                            Weight_dict[(up_type,down_type)][i][j] = Connection_dict[(neuron_name_i,neuron_name_j)][2]
        print("Prepared weight matrix")

        for up_type, down_type in Weight_dict:
            print(up_type,down_type)
            sns.heatmap(Weight_dict[(up_type,down_type)])
            plt.show()
            self.add_connection(source=up_type, target=down_type,
                                weight_list=Weight_dict[(up_type,down_type)].flatten().tolist(),
                                receptor='AMPA',connection_prob=1,model='w:siemens')

        print("Input weight matrix into networl")

        for i,j in Connection_dict:
            up_model, down_model = Neuron_dict[i], Neuron_dict[j]
            up_id = Neuron_model_dict[up_model].index(i)
            down_id = Neuron_model_dict[down_model].index(j)
            connection_id = up_id * len(Neuron_model_dict[down_model]) + down_id
            r,s,ws = Connection_dict[(i,j)]
            model, pre_eq, post_eq = self.Synape_model_dict[(s, r)]
            self.set_connection_attribute((up_model,down_model),connectionids=[connection_id],attribute='model',value=model)
            self.set_connection_attribute((up_model,down_model),connectionids=[connection_id],attribute='on_pre',value=pre_eq)
            self.set_connection_attribute((up_model,down_model),connectionids=[connection_id],attribute='on_post',value=post_eq)
        self.Neuron_model_dict = Neuron_model_dict
        self.Neuron_dict = Neuron_dict
        return Connection_dict

        # model, pre_eq, post_eq = self.Synape_model_dict[(s, r)]
        # self.add_connection(source, target, 1, receptor=r, weight_mean=weight,
        #                     model=model, pre_eq=pre_eq, post_eq=post_eq)

    def load_connections_from_EM(self, file_name, upstream_neuron_list=[], downstream_neuron_list=[],
                                 weight_threshold=0, neuron_type_dict={}, synapse_type_dict={}, up_down_intersection=True):
        '''
        :param file_name:
        :param upstream_neuron_list: if not specified, then choose all neurons
        :param downstream_neuron_list: if not specified, then choose all neurons
        :param weight_threshold: if not specified, then choose zero
        :param neuron_type_dict: if not specified, all neurons will be set as the general LIF model. to specify the neuron type,
                                    for example, you should give neuron_instance: 'hSTDP'
        :param synapse_type_dict: if not specified, all connections will be set as the general synapse model. to specify the connection type,
                                    for example, you should give (source_instance, target_instance): ('hSTDP', 'AMPA',1)
                                    The three variable is synapse model, receptor, weight_scaling_factor
        :return:
        '''
        self.upstream_of_neuron_dict = {}
        self.downstream_of_neuron_dict = {}
        data = pd.read_excel(file_name).values.tolist()
        Neuron_dict = {}
        for connection in data:
            discard, up_id, up_instance, up_type,\
                down_id,down_instance,down_type,weight, roi = connection
            if weight < weight_threshold:
                continue
            if upstream_neuron_list or downstream_neuron_list:
                if not up_down_intersection:
                    if up_type not in upstream_neuron_list and down_type not in downstream_neuron_list:
                        continue
                else:
                    if up_type not in upstream_neuron_list or down_type not in downstream_neuron_list:
                        continue
            source = f'{up_instance}_{up_id}'.replace(">","_to_").replace("(","_").replace(")","_").replace("'","p").replace("/","_and_").replace("-","")
            target = f'{down_instance}_{down_id}'.replace(">","_to_").replace("(","_").replace(")","_").replace("'","p").replace("/","_and_").replace("-","")
            if source not in Neuron_dict:
                if up_type not in neuron_type_dict:
                    self.add_LIF_neuron(source,1, LIF_neuron.eqs)
                    Neuron_dict[source] = 'LIF'
                elif neuron_type_dict[up_type] == 'hSTDP':
                    self.add_hSTDP_neuron(source,1,hSTDP_neuron.eqs)
                    Neuron_dict[source] = 'hSTDP'
                else:
                    raise BaseException(f"Currently doesn't support {neuron_type_dict[up_type]} neuron model!")
                print(f"Add {source} into network")
            if target not in Neuron_dict:
                if down_type not in neuron_type_dict:
                    self.add_LIF_neuron(target,1, LIF_neuron.eqs)
                    Neuron_dict[target] = 'LIF'
                elif neuron_type_dict[down_type] == 'hSTDP':
                    self.add_hSTDP_neuron(target,1,hSTDP_neuron.eqs)
                    Neuron_dict[target] = 'hSTDP'
                else:
                    raise BaseException(f"Currently doesn't support {neuron_type_dict[down_type]} neuron model!")
                print(f"Add {target} into network")

            if (up_type,down_type) not in synapse_type_dict:
                r = 'AMPA'
                s = 'general'
                model, pre_eq, post_eq = self.Synape_model_dict[(s, r)]
                self.add_connection(source,target,1,receptor=r,weight_mean=weight,
                                    model=model,pre_eq=pre_eq, post_eq=post_eq)
            elif synapse_type_dict[(up_type,down_type)][0] == 'general':
                s, r, ws = synapse_type_dict[(up_type,down_type)]
                model, pre_eq, post_eq = Synape_model_dict[(s, r)]
                self.add_connection(source,target,1,receptor=r, weight_mean=weight, weight_scalar=ws,
                                    model=model, pre_eq=pre_eq, post_eq=post_eq)
            elif synapse_type_dict[(up_type,down_type)][0] == 'hSTDP_post':
                raise BaseException("Under development!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if target not in self.upstream_of_neuron_dict:
                self.upstream_of_neuron_dict[target] = []
            self.upstream_of_neuron_dict[target].append(source)

            if source not in self.downstream_of_neuron_dict:
                self.downstream_of_neuron_dict[source] = []
            self.downstream_of_neuron_dict[source].append(target)
            print(f"Add {source} to {target} {r},{s}")

    def construct_connections_random(self, source, target, connection_prob, weight_mean=1,
                                     weight_std=0, weight_scalar=1, normalization_input=False,
                                     normalization_output=False):
        upstream_number = len(self.Connections[(source, target)].source)
        downstream_number = len(self.Connections[(source, target)].target)
        connection_matrix = np.zeros((upstream_number, downstream_number), dtype=float)
        weight_index = np.random.random(upstream_number*downstream_number)
        connection_matrix = np.random.normal(weight_mean, weight_std, upstream_number*downstream_number)
        weight_index[weight_index > connection_prob] = 0
        connection_matrix[np.where(weight_index == 0)] = 0
        connection_matrix[connection_matrix<0] = 0
        connection_matrix = connection_matrix.reshape((upstream_number,downstream_number))
        column_sum = connection_matrix.sum(axis=0)
        for col_id in range(connection_matrix.shape[1]):
            if column_sum[col_id] == 0:
                row_id = rd.choice([i for i in range(connection_matrix.shape[0])])
                connection_matrix[row_id][col_id] = np.random.normal(weight_mean, weight_std)
        column_sum = connection_matrix.sum(axis=0)
        row_sum = connection_matrix.sum(axis=1)
        for row_id in range(connection_matrix.shape[0]):
            if row_sum[row_id] == 0:
                col_id = rd.choice([i for i in range(connection_matrix.shape[1])])
                connection_matrix[row_id][col_id] = np.random.normal(weight_mean, weight_std)
        row_sum = connection_matrix.sum(axis=1)
        if normalization_input:
            connection_matrix = connection_matrix / column_sum * weight_scalar
        elif normalization_output:
            connection_matrix = connection_matrix.transpose()
            connection_matrix = connection_matrix / connection_matrix.sum(axis=0) * weight_scalar
            connection_matrix = connection_matrix.transpose()
        else:
            connection_matrix = connection_matrix * weight_scalar
        self.Connections[(source, target)].connect()
        self.Connections[(source, target)].w = connection_matrix.flatten()*nS
        return

    def neuron_initialization(self, neuron_model, neuron_instance):
        if neuron_model == 'LIF':
            for attribute, value in LIF_neuron.__dict__.items():
                if '__' not in attribute and 'eqs' not in attribute and 'neuron_model' not in attribute:
                    self.set_neuron_attribute(neuron_instance, attribute, value)
        elif neuron_model == 'hSTDP':
            for attribute, value in hSTDP_neuron.__dict__.items():
                if '__' not in attribute and 'eqs' not in attribute and 'neuron_model' not in attribute:
                    self.set_neuron_attribute(neuron_instance, attribute, value)

    def neuron_initialization_general(self, model, neuron_instance):
        for attribute, value in model.__dict__.items():
            if '__' not in attribute and 'eqs' not in attribute and 'neuron_model' not in attribute:
                self.set_neuron_attribute(neuron_instance, attribute, value)

    def connection_initialization(self, connection_instance, synapse):
        for attribute, value in synapse.__dict__.items():
            if '__' not in attribute and 'eqs' not in attribute and 'model' not in attribute:
                self.set_connection_attribute(connection_instance, attribute, value)

    def add_spike_generator_neuron(self, neuron_instance, number, strength, generator_type='poisson'):
        self.spike_generator_dict[neuron_instance] = 'poisson'
        self.Neurons[neuron_instance] = PoissonGroup(N=number, rates=strength,name=neuron_instance)


    def add_neuron(self, neuron_instance, number, model=neuron(), method='euler',
                   threshold=-40, reset=-60, refractory=1 * ms,
                   rest=-70 * mV, x_reset=1, taux=15):
        self.Neuron_instance_dict[neuron_instance] = model.neuron_model
        self.Neurons[neuron_instance] = NeuronGroup(N=number, model=model.eqs, method=method, threshold=f'v>={threshold}*mV',
                                                    reset=f'v={reset}*mV;x_trace+={x_reset}/({taux})',
                                                    refractory=refractory, name=neuron_instance)
        #print("self.Neurons[neuron_instance].I_inject = ",self.Neurons[neuron_instance].I_inject)
        self.Neurons[neuron_instance].v = rest
        self.neuron_initialization_general(model, neuron_instance)


    def add_LIF_neuron(self, neuron_instance, number, eqs, method='euler', threshold=-40, reset=-60, refractory=1 * ms,
                       rest=-70 * mV, x_reset=1, taux=15):
        self.Neuron_instance_dict[neuron_instance] = 'LIF'
        self.Neurons[neuron_instance] = NeuronGroup(N=number, model=eqs, method=method, threshold=f'v>={threshold}*mV',
                                                    reset=f'v={reset}*mV;x_trace+={x_reset}/({taux})',
                                                    refractory=refractory, name=neuron_instance)
        self.Neurons[neuron_instance].v = rest
        self.neuron_initialization('LIF', neuron_instance)

    def add_hSTDP_neuron(self, neuron_instance, number, eqs, method='euler', threshold=-40, reset=-60, refractory=1 * ms,
                         rest=-70 * mV, x_reset=1, taux=15):
        self.Neuron_instance_dict[neuron_instance] = 'hSTDP'
        self.Neurons[neuron_instance] = NeuronGroup(N=number, model=eqs, method=method,
                                                    threshold=f'v>={threshold}*mV',
                                                    reset=f'v={reset}*mV;x_trace+={x_reset}/({taux})',
                                                    refractory=refractory, name=neuron_instance)
        self.Neurons[neuron_instance].v = rest
        self.neuron_initialization('hSTDP', neuron_instance)

    def add_connection(self, source, target, connection_prob=1, synapse=general_synapses(receptor='AMPA'), normalization=False,
                       weight_mean=1.0, weight_std=0, weight_scalar=1, weight_list=[],
                       magic_term=''):
        self.Connections[(source, target)] = Synapses(self.Neurons[source], self.Neurons[target], model=synapse.syn_model,
                                                      on_pre=synapse.pre_eqs, on_post=synapse.post_eqs,name = f"{source}to{target}")

        # if magic_term:
        #     self.Connections[(source, target)].connect()
        #     if magic_term == "i!=j":

        if len(weight_list)!=0:
            self.Connections[(source, target)].connect()
            self.Connections[(source, target)].w = weight_list * nS
            ## 這裡可能要加normalization的狀況
        elif normalization == 'input':
            self.construct_connections_random(
                source, target, connection_prob, weight_mean, weight_std, weight_scalar, normalization_input=True)
        elif normalization == 'output':
            self.construct_connections_random(
                source, target, connection_prob, weight_mean, weight_std, weight_scalar, normalization_output=True)
        elif weight_std != 0:
            self.construct_connections_random(
                source, target, connection_prob, weight_mean, weight_std, weight_scalar)
        else:
            self.Connections[(source, target)].connect(p=connection_prob)
            self.Connections[(source, target)].w = weight_mean * weight_scalar * nS
        self.Connection_copy[(source, target)] = self.Connections[(source, target)].w
        self.connection_initialization((source,target),synapse)
        if 'w_max' in synapse.__dict__.keys() and len(weight_list)!=0:
            N = self.Neurons[source].N*self.Neurons[target].N
            tmp_w_max = []
            for i in range(N):
                if weight_list[i] == 0:
                    tmp_w_max.append(0 * nS)
                else:
                    tmp_w_max.append(synapse.w_max)
            self.Connections[(source, target)].w_max = tmp_w_max





    def set_connection_attribute(self, connection_instance, attribute, value, connectionids=[]):
        try:
            if connectionids:
                for i in connectionids:
                    setattr(self.Connections[connection_instance][i], attribute, value)  # setattr(someobject,foostring, value)
            else:
                setattr(self.Connections[connection_instance], attribute, value)  # setattr(someobject,foostring, value)
        except:
            raise BaseException(
                f'Model for {connection_instance} does not have the attribute: {attribute} or the unit is wrong')

    def set_neuron_attribute(self, neuron_instance, attribute, value, neuronids=[]):
        try:
            if neuronids:
                for i in neuronids:
                    setattr(self.Neurons[neuron_instance][i], attribute, value)  # setattr(someobject,foostring, value)
            else:
                setattr(self.Neurons[neuron_instance], attribute, value)  # setattr(someobject,foostring, value)
        except:
            raise BaseException(
                f'Model for {neuron_instance} does not have the attribute: {attribute} or the unit is wrong')

    def visualise_connectivity(self, source, target, point_line=True, heatmap=True):
        Ns = len(self.Connections[(source, target)].source)
        Nt = len(self.Connections[(source, target)].target)

        if point_line:
            plt.figure(figsize=(10, 4))
            plot([0, 1], [self.Connections[(source, target)].i - Ns / 2, self.Connections[(source, target)].j - Nt / 2],
                 'k.')
            for i in range(Ns):
                for j in range(Nt):
                    w = self.Connections[(source, target)].w[i * Nt + j]/nS
                    if w > 0:
                        plot([0, 1], [i - Ns / 2, j - Nt / 2], '-k', linewidth=np.log((w+1))/2)
            xticks([0, 1], ['Source', 'Target'])
            ylabel('Neuron index')
            xlim(-0.1, 1.1)
            plt.show()

        if heatmap:
            W = np.reshape(self.Connections[(source, target)].w, (Ns, Nt))/nS
            sns.heatmap(W)
            plt.xlabel("Downstream neuron")
            plt.ylabel("Upstream neuron")
            plt.show()
            try:
                sns.clustermap(W)
                plt.xlabel("Downstream neuron")
                plt.ylabel("Upstream neuron")
                plt.show()
            except:
                print("Not enough number to do clustermap")
                pass

class Simulation(Stimulation_generator, Configuration_generator):
    def __init__(self, dt=1, eNose_data_loding=True):
        self.pca_bin_edges_dict = {}
        self.pca_bin_width_dict = {}
        self.event_number = 0
        self.Monitors = {}
        self.current_protocols = {}
        self.Stimulation_firing_rate_dict = {}
        self.Firing_rate_transformed_dict = {}
        self.network_op = NetworkOperation(self.update_current_injection, dt=dt * ms)
        self.net = Network()
        self.dt = dt
        self.Neurons = {}
        self.Neuron_instance_dict = {}
        self.spike_generator_dict = {}
        self.Connections = {}
        self.Connection_copy = {}
        self.Protocols = {}
        self.timeline = []
        self.pooled_max_dict = {}
        self.multi_set = False
        self.AMPA_synapse = general_synapses(receptor='AMPA')
        self.GABA_synapse = general_synapses(receptor='GABA')
        self.Ach_synapse = general_synapses(receptor='Ach')
        self.Synape_model_dict = {('general', 'AMPA'): [self.AMPA_synapse.syn_model, self.AMPA_synapse.pre_eqs,
                                                        self.AMPA_synapse.post_eqs],
                                  ('general', 'GABA'): [self.GABA_synapse.syn_model, self.GABA_synapse.pre_eqs,
                                                        self.GABA_synapse.post_eqs],
                                  ('general', 'Ach'): [self.Ach_synapse.syn_model, self.Ach_synapse.pre_eqs,
                                                       self.Ach_synapse.post_eqs]
                                  }
        if eNose_data_loding:
            try:
                self.load_eNose()
            except:
                raise BaseException("You are trying to load eNose data. "
                                    "However, it fails. Or change the variable 'eNose_data_loading as False")

    def update_current_injection(self, t):
        time_index = int(int(t / ms) / self.dt) ## current time a.u.
        for protocol_index, protocol in enumerate(self.timeline):
            if time_index < protocol[1]: ## not start yet
                break
            if time_index > protocol[2]: ## finished
                continue
            elif time_index == protocol[2]: ## end
                if protocol[-2] == 'I_inject':
                    neuron_name = protocol[-1]
                    self.net[neuron_name].I_inject = 0 * amp
                elif protocol[-2] == 'active':
                    if neuron_name in self.spike_generator_dict:
                        neuron_name = protocol[-1]
                        protocol_name = protocol[0]
                        self.net[neuron_name].active = False
                    else:
                        print("ERROR!! The neuron is not spike generator!!!")
            elif protocol[2]> time_index >= protocol[1]: ## during the protocol
                if protocol[-2] == 'I_inject':
                    neuron_name = protocol[-1]
                    protocol_name = protocol[0]
                    start_time = protocol[1]
                    self.net[neuron_name].I_inject = self.Protocols[protocol_name][time_index-start_time]
                    print("self.net[neuron_name].I_inject = ",self.net[neuron_name].I_inject)
                elif protocol[-2] == 'active':
                    if neuron_name in self.spike_generator_dict:
                        neuron_name = protocol[-1]
                        protocol_name = protocol[0]
                        self.net[neuron_name].active = True
                    else:
                        print("ERROR!! The neuron is not spike generator!!!")

                elif protocol[-2] == 'w_output_normalization':
                    source, target = protocol[-1]
                    source_num = len(self.net[source].I_inject)
                    target_num = len(self.net[target].I_inject)
                    connection_name = f"{source}to{target}"
                    w = np.asarray(self.net[connection_name].w/nS).reshape(source_num,target_num).transpose()
                    w = w/w.sum(axis=0)
                    self.net[connection_name].w = w.transpose().ravel()*self.net[connection_name].w_max
                elif protocol[-2] == 'w_input_normalization':
                    source, target = protocol[-1]
                    source_num = len(self.net[source].I_inject)
                    target_num = len(self.net[target].I_inject)
                    connection_name = f"{source}to{target}"
                    w = np.asarray(self.net[connection_name].w/nS).reshape(source_num,target_num)
                    w = w/w.sum(axis=0)
                    self.net[connection_name].w = w.ravel()*self.net[connection_name].w_max

    def run(self, runtime=0):
        if runtime == 0:
            for neuron_instance in self.spike_generator_dict:
                self.Neurons[neuron].active = False
            runtime = max([i[2] for i in self.timeline])+10
            runtime = runtime * ms

        self.timeline = sorted(self.timeline, key = lambda k: k[1])
        for neuron in self.Neurons:
            self.net.add(self.Neurons[neuron])
        for monitor in self.Monitors:
            self.net.add(self.Monitors[monitor])
        for connection in self.Connections:
            self.net.add(self.Connections[connection])
        self.net.add(self.network_op)
        self.net.run(runtime)

    def add_neuron_state_monitor(self, neuron_instance, variable):
        '''

        :param neuron_instance:
        :param variable:
        :return:
        '''
        if variable != 'spike':
            self.Monitors[neuron_instance, variable] = StateMonitor(self.Neurons[neuron_instance], variable,
                                                                        record=True)
        else:
            self.Monitors[neuron_instance, variable] = SpikeMonitor(self.Neurons[neuron_instance])

    def add_connection_state_monitor(self, source, target, variables='v'):
        self.Monitors[(source, target),variables] = StateMonitor(self.Connections[(source, target)],
                                                     variables=variables, record=True)

    def analyze_firing_rate_pca(self, neuron_instance, n_components=3, show_fig=True, file_name=''):
        all_data = []
        index_collection = []
        stimulation_firing_rate_dict = self.Stimulation_firing_rate_dict[neuron_instance]
        colors = distinctipy.get_colors(len(stimulation_firing_rate_dict))
        for odor in stimulation_firing_rate_dict:
            start = len(all_data)
            all_data += stimulation_firing_rate_dict[odor]
            end = len(all_data)
            index_collection.append((start, end, odor))
        firing_rate_transformed_dict = {}
        # print("######")
        # print(all_data)
        # print("######")
        all_data = np.array(all_data)
        pca = PCA(n_components=n_components)
        pca.fit(all_data)
        all_data = pca.transform(all_data)
        for i in range(len(index_collection)):
            start, end, odor = index_collection[i]
            firing_rate_transformed_dict[odor] = all_data[start:end]
        self.Firing_rate_transformed_dict[neuron_instance] = firing_rate_transformed_dict
        '''
        data format is [[pca1 pca2 pca3],[pca1 pca2 pca3], ....]
        '''

        if show_fig or file_name:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for labelindex, label in enumerate(index_collection):
                plt.plot(all_data[label[0]:label[1], 0], all_data[label[0]:label[1], 1], all_data[label[0]:label[1], 2],
                         '.', color=colors[labelindex], label=label[2], markersize=10)
            plt.legend()
            plt.title(neuron_instance)
            plt.show()

    def plot_firing_rate(self,neuron_instance, neuronids=[], all=True, show_fig=True, file_name='',start_time=0,time_window=20, end_time=0, protocol_target='PN'):
        if end_time == 0:
            end_time = self.Current_T
        time_list = self.Monitors[neuron_instance, 'spike'].t ### time_list <spikemonitor.t: array([0.1095, 0.1118, 0.1119, ..., 3.3986, 3.3992, 3.3995]) * second>
        print("time_list",len(time_list),end_time)
        spikes_list = self.Monitors[neuron_instance, 'spike'].i
        tmp_list = (time_list / second * 1000).astype('int') ## change to ms but get floor to int
        print('tmp_list, spike_list',len(tmp_list),len(spikes_list))
        tmp_spike_list = [[0 for _ in range(end_time)] for __ in range(self.Neurons[neuron_instance].N)]
        firing_rate_list = [[0 for _ in range(end_time)] for __ in range(self.Neurons[neuron_instance].N)]
        for t,nid in zip(tmp_list,spikes_list):
            tmp_spike_list[nid][t] += 1
        print(len(tmp_spike_list))
        for nid in range(self.Neurons[neuron_instance].N):
            for t in range(end_time):
                for temp in range(-time_window, time_window):
                    if t + temp < start_time:
                        continue
                    elif t + temp >= end_time:
                        continue
                    firing_rate_list[nid][t] += tmp_spike_list[nid][t + temp]
        data = np.array(firing_rate_list,dtype=float)/(time_window*2)*1000
        stimulation_firing_rate_dict = {}
        for protocol in self.timeline:
            if protocol[1] < start_time or protocol[2] > end_time:
                continue
            if protocol[-1] != protocol_target:
                continue
            stimulation_type = protocol[-3] ## odor type
            if stimulation_type not in stimulation_firing_rate_dict:
                stimulation_firing_rate_dict[stimulation_type] = []
            tmp = []
            for i in range(self.Neurons[neuron_instance].N):
                if i not in neuronids and not all:
                    continue
                max_firing_rate = max(data[i][protocol[1]:protocol[2]])
                tmp.append(max_firing_rate)
            stimulation_firing_rate_dict[stimulation_type].append(tmp)
        self.Stimulation_firing_rate_dict[neuron_instance] = stimulation_firing_rate_dict
        # data = data [:,start_time:end_time]
        if show_fig or file_name:
            for i in range(self.Neurons[neuron_instance].N):
                if i not in neuronids and not all:
                    continue
                plt.plot([t for t in range(data.shape[1])], data[i])
            plt.xlim((int(start_time), int(end_time)))
            plt.xlabel("time (ms)")
            plt.ylabel("Firing rate (Hz)")
            plt.title(neuron_instance)
            if file_name:
                plt.savefig(file_name)
            if show_fig:
                plt.show()
            plt.close()

        return stimulation_firing_rate_dict


    def plot_spikes(self, neuron_instance, neuronids=[], all=True, show_fig=True, file_name='',start_time=0, end_time=-1):
        time_list = self.Monitors[neuron_instance, 'spike'].t
        if len(time_list) == 0:
            if show_fig or file_name:
                plt.plot([0], [0])
                plt.title("Raster plot")
                plt.ylabel(f"{neuron_instance} index")
                plt.xlabel("Time(s)")
                plt.tight_layout()
            if file_name:
                plt.savefig(file_name)
            if show_fig:
                plt.show()
            plt.close()
            return
        spikes_list = self.Monitors[neuron_instance, 'spike'].i
        start_time = start_time * ms
        if end_time < 0:
            end_time = time_list[-1]
        else:
            end_time = end_time * ms
        # plt.close()
        if neuronids and not all:
            data = [[i,j] for i,j in zip(time_list,spikes_list) if spikes_list[j] in neuronids if i >= start_time and i <= end_time]
            data = np.array(data)
            time_list = data[:,0]
            spikes_list = data[:,1]
        else:
            data = [[i,j] for i,j in zip(time_list,spikes_list) if i >= start_time and i <= end_time]
            data = np.array(data)
            time_list = data[:,0]
            spikes_list = data[:,1]
        if show_fig or file_name:
            f = plt.figure()
            plt.plot(time_list, spikes_list, 'k.')
            plt.title("Raster plot")
            plt.ylabel(f"{neuron_instance} index")
            plt.xlabel("Time(s)")
            plt.tight_layout()
        if file_name:
            plt.savefig(file_name)
        if show_fig:
            plt.show()
        plt.close()
        return data

    def plot_variable(self,monitored_instance, variable, ids=[], all=True, show_fig=True, file_name='', start_time=0, end_time=-1, time_window=5, show_label=False):
        '''
        All the monitored variables can be plot here.
        If you want to specify the neuron id, you can use ids list and change all from True to False.
        Also you can limit the display time.
        time_window currently is used when you want to calculate firing rate!
        :param monitored_instance: neuron instance
        :param variable: attribute of the neuron
        :param ids: specify neuron_id_list or the default will display all
        :param all:
        :param show_fig:
        :param file_name:
        :param start_time:
        :param end_time:
        :param time_window:
        :return:
        '''
        if ids:
            all = False
        if (monitored_instance, variable) not in self.Monitors:
            print('You have not set the monitor for the variable. Please set the monitor before you run simulation.')
            return
        elif variable == 'spike':
            data = self.plot_spikes(monitored_instance,neuronids=ids,all=all,
                             show_fig=show_fig,file_name=file_name,start_time=start_time, end_time=end_time)
            ## data = [[time_1, neuron_id],[time_2, neuron_id],....]
            return data
        elif variable == 'firing rate':
            data = self.plot_firing_rate(monitored_instance,ids,all,show_fig,file_name,start_time,time_window,end_time)
            return data
        time_list = self.Monitors[monitored_instance, variable].t
        if end_time < 0:
            end_time = time_list[-1]
        else:
            end_time = end_time * ms
        value = np.asarray(self.Monitors[monitored_instance, variable].state(variable))
        value = np.array([value[i] for i in range(value.shape[0]) if time_list[i] >= start_time and time_list[i] <= end_time])
        value = value.transpose()
        time_list = np.array([time_list[i] for i in range(value.shape[1]) if time_list[i] >= start_time and time_list[i] <= end_time])
        if show_fig or file_name:
            if ids and not all:
                for id in ids:
                    plot(time_list / ms, value[id], label=f"{id}")
            else:
                for id in range(value.shape[0]):
                    plot(time_list / ms, value[id], label=f"{id}")
            plt.ylabel(f"{monitored_instance}_{variable}")
            plt.xlabel("Time(s)")
            if show_label:
                plt.legend()
            plt.tight_layout()

        if file_name:
            plt.savefig(file_name)
            plt.close()
        if show_fig:
            plt.show()

        return value

    def estimate_accuracy(self, stimulated_instance='PN', answer_instance='MBON',time_window=20):
        start_time = 0
        end_time = self.Current_T
        time_list = self.Monitors[
            answer_instance, 'spike'].t  ### time_list <spikemonitor.t: array([0.1095, 0.1118, 0.1119, ..., 3.3986, 3.3992, 3.3995]) * second>
        # print("time_list",time_list)
        spikes_list = self.Monitors[answer_instance, 'spike'].i
        tmp_list = (time_list / second * 1000).astype('int')  ## change to ms but get floor to int
        # print(tmp_list)
        tmp_spike_list = [[0 for _ in range(end_time)] for __ in range(self.Neurons[answer_instance].N)]
        firing_rate_list = [[0 for _ in range(end_time)] for __ in range(self.Neurons[answer_instance].N)]
        for t, nid in zip(tmp_list, spikes_list):
            tmp_spike_list[nid][t] += 1
        # print(tmp_spike_list)
        for nid in range(self.Neurons[answer_instance].N):
            for t in range(end_time):
                for temp in range(-time_window, time_window):
                    if t + temp < start_time:
                        continue
                    elif t + temp >= end_time:
                        continue
                    firing_rate_list[nid][t] += tmp_spike_list[nid][t + temp]
        data = np.array(firing_rate_list, dtype=float) / (time_window * 2) * 1000
        Answer_sheet = []
        stimulation_firing_rate_dict = {}
        for protocol in self.timeline:
            if 'test' not in protocol[0] or 'PN' not in stimulated_instance:
                continue
            stimulation_type = protocol[-3] ## odor type
            print(protocol)
            if stimulation_type not in stimulation_firing_rate_dict:
                stimulation_firing_rate_dict[stimulation_type] = []
            tmp = []
            for i in range(self.Neurons[answer_instance].N):
                max_firing_rate = max(data[i][protocol[1]:protocol[2]])
                tmp.append(max_firing_rate)
            max_response_neuron_id = np.argmax(tmp)
            if max_response_neuron_id == self.target_odor_list.index(stimulation_type):
                Answer_sheet.append(1)
            else:
                Answer_sheet.append(0)
            stimulation_firing_rate_dict[stimulation_type].append(tmp)
        Accuracy = float(np.sum(Answer_sheet))/len(Answer_sheet)
        print(f"Accuracy: {Accuracy}")
        return Accuracy


if __name__ == '__main__':
    exp = Simulation()
    exp.load_eNose()
    exp.get_odor_stimulation_PCA(visualization=True)