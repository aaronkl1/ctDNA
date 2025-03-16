import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt

class BirthDeath:
    """
    simulate the birth-death process using Gillespie algorithm.
    """
    
    def __init__ (self, params):
        """
        initialize the population.
        inputs:
        birth_rate: float, birth rate per individual.
        death_rate: float, death rate per individual.
        N0: int, initial population size.
        """
        self.s_birth_rate = params['s_birth_rate']
        self.s_death_rate = params['s_death_rate']
        self.s_pop_size = params['s_N']
        self.s_pop_hist = [self.s_pop_size]    # list to record history of population size
        self.r_birth_rate = params['r_birth_rate']
        self.r_death_rate = params['r_death_rate']
        self.r_pop_size = params['r_N']
        self.r_pop_hist = [self.r_pop_size]
        self.total_pop_hist = [self.s_pop_size + self.r_pop_size]
        self.elim_rate = params['elim_rate']
        self.shed_prob = params['shed_prob']
        self.drug = params['drug']
        self.drug_hist = [0]
           # current population size
        self.ctDNA = 0 #amount of ctDNA
        self.time = 0.    # time since beginning of simulation
        self.ctDNA_hist = [0] #list to record ctDNA amount 
        self.time_hist = [0.]    # list to record time of all events
        self.detected = False
        self.detection_time = 100
        # self.detection_index = -1
        # self.drug_start_index = -1
        self.min_size = float('inf')
        # self.baseline_ctDNA = -1
        # self.peak = -1
        # self.halfhour = -1
        # self.peaktime = 0
        # self.twohour = -1
        self.drug_start_s_prop = -1
        self.drug_start_size = 10000000
        self.samples = params['sampling'][0]
        self.window = params['sampling'][1]
        self.pre_samples = [-1]*self.samples
        self.i = -1
        self.post_samples = [-1]*self.samples   


    def drugConc(self):
        first, period, k1, k2, A, _, _ = self.drug
        conc = 0
        while first < self.time:
            conc+=k1*A*(np.exp(-k1*(self.time-first)) - np.exp(-k2*(self.time-first)))/(k2-k1)
            first+=period
        return conc

    
    def next_event(self):
        """
        generate the waiting time and identity of the next event.
        outputs:
        tau: float, waiting time before next event.
        event: int, 0 means birth and 1 means death.
        """
        s_b = self.s_birth_rate
        s_d = self.s_death_rate
        r_b = self.r_birth_rate
        r_d = self.r_death_rate

        conc = self.drugConc()
        self.drug_hist.append(conc)

        s_b += self.drug[5]*conc/self.drug[4]
        s_d += self.drug[6]*conc/self.drug[4]

        # c = (1- (self.s_pop_size + self.r_pop_size)/(100000))
        c = 1
        s_k_b = self.s_pop_size * s_b*c# total sensitive birth rate
        s_k_d = self.s_pop_size * s_d   # total sensitive death rate
        r_k_b = self.r_pop_size * r_b*c
        r_k_d = self.r_pop_size * r_d   
        k_e = self.ctDNA * self.elim_rate 
        tot = (s_k_b + s_k_d + r_k_b + r_k_d + k_e)
        if tot <= 0:
            # print(self.s_pop_size, self.r_pop_size, self.ctDNA, s_b, s_d, r_b, r_d)
            return 0,5
        tau = np.random.exponential(1/tot)
          # draw a random number from exponential dist as putative death time
        earliest = np.random.rand()
        if earliest < s_k_b/tot:    # birth happens first
            event = 0    # use 0 to label birth
        elif earliest < (s_k_b+s_k_d)/tot:    # death happens first
            event = 1    # use 1 to label death
        elif earliest < (s_k_b+s_k_d+r_k_b)/tot:
            event = 2
        elif earliest < (s_k_b+s_k_d+r_k_b+r_k_d)/tot:
            event = 3
        else:  #elim happens first
            event = 4
        return tau, event

    
    def run(self, T):
        """
        run simulation until time T since the beginning.
        inputs:
        T: float, time since the beginning of the simulation.
        """
        while self.time < T:
        # while self.s_pop_size + self.r_pop_size < 2*self.drug_start_size:
            if self.s_pop_size <= 0 and self.r_pop_size <= 0 and self.ctDNA <= 0:    # population is extinct
                break    # exit while loop to end simulation
            tau, event = self.next_event()    # draw next event
            if self.detected:
                if 0 <= self.i < self.samples and self.time+tau > self.detection_time + self.i*self.window/self.samples:
                    self.pre_samples[self.i] = self.ctDNA
                    self.i+=1
                elif self.samples <= self.i < 2*self.samples and self.time+tau > self.detection_time + self.i*self.window/self.samples:
                    self.post_samples[self.i-self.samples] = self.ctDNA
                    self.i+=1
            # if self.time > self.detection_time + 2 and self.two

            self.time += tau    # update time
            if event == 0:    # birth happens
                self.s_pop_size += 1    # increase population size by 1
            elif event == 1:    # death happens
                self.s_pop_size -= 1    # decrease population size by 1
                if np.random.rand() < self.shed_prob : 
                    # * 2000/(self.s_pop_size + self.r_pop_size)**(1/3)
                    # self.ctDNA+=10
                    self.ctDNA+=1
            elif event == 2:    # birth happens
                self.r_pop_size += 1    # increase population size by 1
            elif event == 3:    # death happens
                self.r_pop_size -= 1    # decrease population size by 1
                if np.random.rand() < self.shed_prob: 
                    self.ctDNA+=1
            elif event == 4:
                self.ctDNA-=1 
            else:
                print(self.s_pop_size, self.r_pop_size, self.ctDNA, self.s_birth_rate, self.s_death_rate, self.r_birth_rate, self.r_death_rate)
                break
            self.time_hist.append(self.time)    # record time of event
            self.s_pop_hist.append(self.s_pop_size)    # record population size after event
            self.r_pop_hist.append(self.r_pop_size) 
            self.total_pop_hist.append(self.s_pop_size + self.r_pop_size)
            self.ctDNA_hist.append(self.ctDNA)

params1 = {
    's_birth_rate': 1.1,   # birth rate
    's_death_rate': 1,    # death rate
    's_N': 9000,   #initial pop
    'r_birth_rate': 1.1,   # birth rate
    'r_death_rate': 1,    # death rate
    'r_N': 1000,   #initial pop
    'elim_rate': 33., #elimination rate
    'shed_prob': 1,  #shedding probability
    'drug': [2, 1, 10, 1, 40, 0, 1], #first, period, k1, k2, A, eff on b, eff on d
    'label': 'cytotoxic',
    'sampling': [5,0.5]
}



def runSim(params, T):
    bd = BirthDeath(params)
    bd.run(T)
        # name of csv file 
    # filename = 'biomarkers/sim_data/'+ 'cytotoxic-' +  '20%eff' ".csv"
        
    # # writing to csv file 
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerows([bd.time_hist, bd.s_pop_hist, bd.r_pop_hist, bd.total_pop_hist, bd.ctDNA_hist])
    fig, axs = plt.subplots(2,1, sharex = 'all')
    #, figsize = (6,7)
    # plt.xticks(visible=False)
    # for a in axs:
    #     plt.setp(a.get_yticklabels(), visible=False)
    
    axs[0].set_ylabel('tumor population')
    axs[0].plot(bd.time_hist, bd.s_pop_hist, label = 'sensitive')
    axs[0].plot(bd.time_hist, bd.r_pop_hist, label = 'resistant')
    axs[0].plot(bd.time_hist, bd.total_pop_hist, label = 'total', color = 'black')
    # axs[0].plot(bd.time_hist[50000:], bd.total_pop_his t[50000:], label = 'total', color = 'grey')

    col = 'tab:green'
    ax2 = axs[1].twinx()
    ax2.set_ylabel('drug concentration (mg)', color = col)
    ax2.plot(bd.time_hist, bd.drug_hist, color = col)

    axs[0].legend()

    axs[1].set_xlabel('time (days)')
    axs[1].set_ylabel('ctDNA (hGE)', color = 'red')
    axs[1].plot(bd.time_hist, bd.ctDNA_hist, color = 'red')  
    # axs[1].plot([bd.detection_time + i*bd.window/(bd.samples) for i in range(2*bd.samples)], bd.pre_samples+bd.post_samples, color = 'black', linestyle = 'None',marker='o', markersize = 3) 
    # axs[1].set_yscale('log')
    # axs[1].set_ylim(20,700)
    # axs[1].plot(bd.time_hist[50000:], bd.ctDNA_hist[50000:], color = 'lightcoral')   
    # #riediger
    # xdata = np.array([-24.0/24, 2.5/24, 26.0/24, 46.5/24, 69.0/24, 93.0/24, 118.0/24, 146.0/24, 164.0/24], dtype = np.float128)
    # ydata = np.array([225.0, 804.0, 2420.0, 1237.0, 363.0, 119.0, 100.0, 71.0, 10.0])
    # axs[1].plot(xdata, ydata, label = 'data', marker = 'x', linestyle = 'None')
 
    # axs[1].axvline(x = bd.detection_time+bd.window*(bd.samples-1)/bd.samples, color = 'black') 
    plt.tight_layout()
    date = datetime.now().strftime("%m-%d")
    filename =   date + 'test'
    plt.savefig(filename +'.eps', format='eps', transparent = True)
    plt.savefig(filename +'.png', format='png')
    # print(bd.pre_samples, bd.post_samples)
    # print(bd.pre_samples, bd.post_samples)
    plt.show()



def genPatients(T,k,samples, window, filename):
    parameters = [0]*k
    s_prop = [0]*k
    drug_start_s_prop = [0]*k
    drug_start_size = [0]*k
    # updown = [0]*k
    # maxes = [0]*k
    # baseline_ctDNA = [0]*k
    min_tumor_size = [0]*k
    end_size = [0]*k
    # slope = [0]*k
    # two_hour_size = [0]*k
    pre_samples = [0]*k
    post_samples = [0]*k
    sample_times = [0]*k
    i = 0
    while i < k:
        s_b = np.random.uniform(0.1,0.2)
        s_d = np.random.uniform(0.05, 0.08)
        # r_b = np.random.rand()*0.5+1
        # r_d = np.random.rand()*(0.8*r_b-0.5) + 0.5
        s_prop[i] = np.random.uniform(0,1)
        initial_pop = np.random.uniform(200,20000)
        detection_size = initial_pop+ np.random.uniform(2500,5000)
        K = np.random.uniform(0.7, 1)
        params = {
            's_birth_rate': s_b,   # birth rate
            's_death_rate': s_d,    # death rate
            's_N': int(initial_pop*s_prop[i]),   #initial pop
            'r_birth_rate': s_b,   # birth rate
            'r_death_rate': s_d,    # death rate
            'r_N': initial_pop-int(initial_pop*s_prop[i]),   #initial pop
            'elim_rate': 33., #elimination rate
            'shed_prob': 1,  #shedding probability
            'drug': [detection_size, -np.random.uniform(0.0, 0.05),np.random.uniform(0.3, 0.5)], #[detection size, effect on b, effect on d]
            # 'drug': [detection_size, -s_b*K, s_b*K], 
            'sampling':  [samples,window]
        }
        bd = BirthDeath(params)
        bd.run(T)
        parameters[i] = str(params)
        drug_start_s_prop[i] = bd.drug_start_s_prop
        drug_start_size[i] = bd.drug_start_size
            # if bd.peak/bd.baseline_ctDNA >= 1.4 and s_prop[i] < 0.1 or bd.peak/bd.baseline_ctDNA < 1.5 and s_prop[i] > 0.9:
            #     print(params)
            # baseline_ctDNA[i] = bd.baseline_ctDNA
            # updown[i] = (bd.peak-bd.baseline_ctDNA + bd.peak-bd.halfhour)
            # maxes[i] = bd.peak/bd.baseline_ctDNA
        min_tumor_size[i] = bd.min_size
        end_size[i] = bd.s_pop_size + bd.r_pop_size
            # slope[i] = (bd.peak-bd.baseline_ctDNA)/(bd.peaktime - bd.drug_start_time)
            # two_hour_size[i] = bd.twohour/detection_size
        
        pre_samples[i] = bd.pre_samples
        post_samples[i] = bd.post_samples
        sample_times[i] = [bd.detection_time + i*bd.window/(bd.samples) for i in range(2*bd.samples)]
        if min(pre_samples[i]) <= 0.0 or min(post_samples[i]) <= 0.0 or min_tumor_size[i] <= 0.0:
            print(i)
            print(params)
        else:
            i+=1
        if i == k//4 or i == k//2 or i == 3*k//4:
            print(i)
    

        
    # writing to csv file 
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows([parameters,drug_start_s_prop, drug_start_size, min_tumor_size,end_size, pre_samples, post_samples, sample_times])

        # csvwriter.writerows([s_prop, baseline_ctDNA, maxes, updown, slope, min_tumor_size, two_hour_size, drug_start_s_prop])
    # plt.figure()
    # plt.scatter(s_prop, norm_peak_ctDNA)
    # plt.xlabel('Proportion of sensitive cells')
    # plt.ylabel('max ctDNA')
    # # plt.ylim(0,5)

    # # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # # plt.legend()
    # plt.savefig('biomarkers/Figures/' + 'genPatients-' + 'res-vs-max'+'.eps', format='eps')
    # plt.savefig('biomarkers/Figures/' + 'genPatients-' +'res-vs-max'+'.png', format='png')
    # plt.show()


# maxTestAll(params1,0.5,100)
runSim(params1, 10)



