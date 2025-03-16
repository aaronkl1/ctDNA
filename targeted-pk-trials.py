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
        # self.s_pop_hist = [self.s_pop_size]    # list to record history of population size
        self.r_birth_rate = params['r_birth_rate']
        self.r_death_rate = params['r_death_rate']
        self.r_pop_size = params['r_N']
        # self.r_pop_hist = [self.r_pop_size]
        # self.total_pop_hist = [self.s_pop_size + self.r_pop_size]
        self.elim_rate = params['elim_rate']
        self.shed_prob = params['shed_prob']
        self.drug = params['drug']
        self.conc = 0
        # self.drug_hist = [0]
           # current population size
        self.ctDNA = 0 #amount of ctDNA
        self.time = 0.    # time since beginning of simulation
        # self.ctDNA_hist = [0] #list to record ctDNA amount 
        # self.time_hist = [0.]    # list to record time of all events
        self.detected = False
        self.detection_time = 100
        # self.detection_index = -1
        # self.drug_start_index = -1
        self.min_size = float('inf')
        self.flag = True
        self.logistic = params['logistic']


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

        self.conc = self.drugConc()
        # self.drug_hist.append(conc)

        s_b += self.drug[5]*self.conc/self.drug[4]
        s_d += self.drug[6]*self.conc/self.drug[4]
        
        

        # c = (1- (self.s_pop_size + self.r_pop_size)/(100000))
        c = 1
        if self.logistic:
            c = (1- (self.s_pop_size + self.r_pop_size)/(100000))
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

    
    def run(self, T, limit = float('inf'), surface = False):
        start = self.time
        if self.time > 2 and self.r_pop_size+self.s_pop_size < self.min_size:
            self.min_size = self.r_pop_size+self.s_pop_size
        if self.s_pop_size + self.r_pop_size > limit:
            self.flag = False
        """
        run simulation until time T since the beginning.
        inputs:
        T: float, time since the beginning of the simulation.
        """
        while self.time-start < T and self.flag:
        # while self.s_pop_size + self.r_pop_size < 2*self.drug_start_size:
            if self.s_pop_size <= 0 and self.r_pop_size <= 0 and self.ctDNA <= 0:    # population is extinct
                break    # exit while loop to end simulation
            tau, event = self.next_event()    # draw next event


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
                if surface:
                    r = ((self.s_pop_size+self.r_pop_size)/4)**(1/3)
                    surface_prob = r/3
                else:
                    surface_prob = 1
                if np.random.rand() < self.shed_prob*surface_prob: 
                    self.ctDNA+=1
            elif event == 4:
                self.ctDNA-=1 
            else:
                print(self.s_pop_size, self.r_pop_size, self.ctDNA, self.s_birth_rate, self.s_death_rate, self.r_birth_rate, self.r_death_rate)
                break

params1 = {
    's_birth_rate': 1.1,   # birth rate
    's_death_rate': 1,    # death rate
    's_N': 9000,   #initial pop
    'r_birth_rate': 1.1,   # birth rate
    'r_death_rate': 1,    # death rate
    'r_N': 1000,   #initial pop
    'elim_rate': 33., #elimination rate
    'shed_prob': 0.5,  #shedding probability
    'drug': [2, 1, 30, 0.7, 40, 0, 1] #first, period, k1, k2, A, eff on b, eff on d
}



def runSim(params, T, samples):
    incr = T/samples
    time_hist = [0]*samples
    s_hist = [0]*samples
    r_hist = [0]*samples
    total_hist = [0]*samples
    ctDNA_hist = [0]*samples
    conc_hist = [0]*samples

    bd = BirthDeath(params)

    for i in range(samples):
        time_hist[i] = i*incr + 0
        s_hist[i] = bd.s_pop_size
        r_hist[i] = bd.r_pop_size
        total_hist[i] = s_hist[i]+r_hist[i]
        ctDNA_hist[i] = bd.ctDNA
        conc_hist[i] = bd.conc
        bd.run(incr)
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
    axs[0].plot(time_hist,s_hist, label = 'sensitive')
    axs[0].plot(time_hist,r_hist, label = 'resistant')
    axs[0].plot(time_hist,total_hist, label = 'total', color = 'black')
    # axs[0].plot(bd.time_hist[50000:], bd.total_pop_his t[50000:], label = 'total', color = 'grey')

    

    axs[0].legend()

    axs[1].set_xlabel('time')
    axs[1].set_ylabel('ctDNA')
    
    col = 'tab:green'
    ax2 = axs[1].twinx()
    ax2.set_ylabel('drug concentration', color = col)
    ax2.plot(time_hist, conc_hist, color = col)
    axs[1].plot(time_hist, ctDNA_hist, color = 'red')  
    # axs[1].plot([bd.detection_time + i*bd.window/(bd.samples) for i in range(2*bd.samples)], bd.pre_samples+bd.post_samples, color = 'black', linestyle = 'None',marker='o', markersize = 3) 
    # axs[1].set_yscale('log')
    # axs[1].set_ylim(20,700)
    # axs[1].plot(bd.time_hist[50000:], bd.ctDNA_hist[50000:], color = 'lightcoral')   
    # #riediger
    # xdata = np.array([-24.0/24, 2.5/24, 26.0/24, 46.5/24, 69.0/24, 93.0/24, 118.0/24, 146.0/24, 164.0/24], dtype = np.float128)
    # ydata = np.array([225.0, 804.0, 2420.0, 1237.0, 363.0, 119.0, 100.0, 71.0, 10.0])
    # axs[1].plot(xdata, ydata, label = 'data', marker = 'x', linestyle = 'None')
 
    # axs[1].axvline(x = bd.detection_time+bd.window*(bd.samples-1)/bd.samples, color = 'black') 
    date = datetime.now().strftime("%m-%d")
    filename =  date + '90s-10r'
    plt.savefig(filename +'.eps', format='eps', transparent = True)
    plt.savefig(filename +'.png', format='png')
    # print(bd.pre_samples, bd.post_samples)
    # print(bd.pre_samples, bd.post_samples)
    plt.show()



def genPatients(k, filename,surface = False, logistic = False):
    # parameters = [0]*k
    s_prop = [0]*k
    drug_start_s_prop = [0]*k
    drug_start_size = [0]*k
    min_tumor_size = [0]*k
    pre_samples = [0]*k
    post_samples = [0]*k
    i = 0
    while i < k:
        s_b = np.random.uniform(1.1,1.3)
        s_d = np.random.uniform(0.8, 1)
        s_prop[i] = np.random.uniform(0,1)
        initial_pop = np.random.uniform(2000,20000)
        params = {
            's_birth_rate': s_b,   # birth rate
            's_death_rate': s_d,    # death rate
            's_N': int(initial_pop*s_prop[i]),   #initial pop
            'r_birth_rate': s_b,   # birth rate
            'r_death_rate': s_d,    # death rate
            'r_N': initial_pop-int(initial_pop*s_prop[i]),   #initial pop
            'elim_rate': 33., #elimination rate
            'shed_prob': 1,  #shedding probability
            'drug': [2, 1, np.random.uniform(15,30), np.random.uniform(0.5,1.5), 40, 0, 1], #first, period, k1, k2, A, eff on b, eff on d,
            'logistic': logistic
        }
        # parameters[i] = str(params)
        bd = BirthDeath(params)
        bd.run(1)
        pre = [0]*4
        for j in range(4):
            pre[j] = bd.ctDNA
            bd.run(0.25, float('inf'), surface)
        drug_start_s_prop[i] = bd.s_pop_size/(bd.s_pop_size+bd.r_pop_size)
        drug_start_size[i] = bd.s_pop_size+bd.r_pop_size
        
        
        pre_samples[i] = pre
        post = [0]*5
        for j in range(5):
            post[j] = bd.ctDNA
            bd.run(0.25, float('inf'), surface)
        post_samples[i] = post
        bd.run(5, 1.1*drug_start_size[i], surface)
        min_tumor_size[i] = bd.min_size
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
        csvwriter.writerows([drug_start_s_prop, drug_start_size, min_tumor_size, pre_samples, post_samples])

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
# runSim(params1, 10, 1000)
date = datetime.now().strftime("%m-%d")
filename = date + 'pk-sim-patients-logistic' + '.csv'
genPatients(500, filename, False, True)


