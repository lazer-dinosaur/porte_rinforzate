import pandas as pd
import datetime
import time
import numpy as np
import itertools as it
import pylab as p
import timing
from scipy.stats import norm, sem, beta
from print_override import *


def plot(self,μ_,σ_,μ,σ):
    x = np.linspace(0,1,101)
    if self.time_step%50==0:
        # print(self.post_a,a=1)
        # print(self.post_b,a=1)
        f,ax = p.subplots(7)
        colours = ['r','g','b','y','orange']
        print(self.cheat_state.shape,self.time_step)
        for i,(a,c) in enumerate(zip(self.cheat_state[:,self.time_step],colours)):
            ax[0].axvline(a,ls='--',alpha=0.5,lw=3,c=c)
            ax[0].text(a,0,i,rotation=0)
            ax[1].axvline(a,ls='--',alpha=0.5,lw=3,c=c)
            ax[1].text(a,0,i,rotation=0)
            ax[2].axvline(a,ls='--',alpha=0.5,lw=3,c=c)
            ax[2].text(a,0,i,rotation=0)
        for i,c in zip(range(self.num_ban),colours):
            ax[0].plot(self.x_[i],self.pmf_[i],lw=3,c=c,alpha=0.5,label=i)
        for i,(xx,a,b,c) in enumerate(zip(self.x_,self.post_a,self.post_b,colours)):
            ax[1].plot(x,beta.pdf(x,a,b),lw=3,c=c,alpha=0.5,label=i)
        for i,(μμ_,σσ_,c) in enumerate(zip(μ_,σ_,colours)):
            ax[2].plot(x,norm.pdf(x,μμ_,σσ_),lw=3,c=c,alpha=0.5,label=i)
        for i,(μμ,σσ,c) in enumerate(zip(μ,σ,colours)):
            ax[3].plot(x,norm.pdf(x,μμ,σσ),lw=3,c=c,alpha=0.5,label=i)
        # for i,(a,b,c) in enumerate(zip(opponent.post_a,opponent.post_b,colours)):
        #     ax[2].plot(x,beta.pdf(x,a,b),lw=3,c=c,alpha=0.5,label=i)
        ax[0].set_xlim([0,1])
        ax[1].set_xlim([0,1])
        ax[2].set_xlim([0,1])
        ax[3].set_xlim([0,1])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        # p.show()


        cm = colors.ListedColormap(colours)
        bounds=[0,1,2,3,4,5]
        norm11 = colors.BoundaryNorm(bounds, cm.N)
        # img = plt.imshow(zvals, interpolation='nearest', origin='lower',
        #                     cmap=cmap, norm=norm)

        # # make a color bar
        # # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])

        # plt.savefig('redwhite.png')
        # plt.show()

        if self.action_history:
            # f,ax = p.subplots(2)
            ax[4].imshow(np.array(self.action_history)[None,:],aspect='auto',cmap=cm,norm=norm11)
            ax[5].imshow(np.array(self.reward_history)[None,:],aspect='auto')
            μ_hist = np.array(self.μ_hist).T
            σ_hist = np.array(self.σ_hist).T
            # print(σ_hist.shape)
            for μ,σ,c in zip(μ_hist,σ_hist,colours):
                x = range(μ.shape[0])
                # print(μ)
                ax[6].plot(x,μ+σ*3,c=c)
                # ax[6].plot(x,μ+σ,c=c,alpha=0.2)
                # ax[6].plot(x,μ-σ,c=c,alpha=0.2)
        ax[4].set_xlim([0,μ.shape[0]-1])
        ax[5].set_xlim([0,μ.shape[0]-1])
        ax[6].set_xlim([0,μ.shape[0]-1])
        p.tight_layout()
        p.show()

def pring_stuff(self):
    print(bandits.history[:,0],n='orig_truth',a=1)
    print(self.ote,n='ote',a=1)
    print(self.α[0]/(self.α[0]+self.β[0]),n='α β',a=1)
    print((self.ote * self.num_pulls + self.num_pulls_opp * self.α[0] / (self.α[0] + self.β[0])) * 1 ** (self.num_pulls_total) / self.num_pulls_total,n='combine',a=1)
    print()
    print(bandits.history[:,bandits.time_step],n='current_truth',a=1)
    print(self.current_vals,n='current_vals_combined',a=1)
    print(self.post_a/(self.post_a+self.post_b),n='current_post_a_b',a=1)
    # print((self.α[0]/(self.α[0]+self.β[0]) * DECAY ** self.num_pulls_total),n='α β decay',a=1)
    print(opponent.post_a/(opponent.post_a+opponent.post_b),n='current_opp',a=1)
    print()

def summary_junk(B,A1,A2):
    ix = np.argsort(B.history[:,0])[::-1]
    print('sorted',o=1)
    print('true:',o=1)
    print(B.history[:,0][ix],o=1)
    print('A2',o=1)
    print(np.round(A2.ote[ix],2),o=1)
    print('true:',o=1)
    print(np.round(B.history[:,-1][ix],2),o=1)
    print('A2',o=1)
    print(np.round(A2.current_vals[ix],2),o=1)

    # print('true:')
    # print(B.history[:,0][ix])
    print('total pulls',o=1)
    print(A2.num_pulls_total[ix],o=1)
    print('pulls',o=1)
    print(A2.num_pulls[ix],o=1)
    print('opp_pulls',o=1)
    print(A1.num_pulls[ix],o=1)

    print(o=1)
    print(f'num_bandits: {num_bandits}, num_cycles: {num_cycles}, N_trials: {N_trials}',o=1)
    print(o=1)
    print(f'{A1.score} - {A1.label}',o=1)
    print(f'{A2.score} - {A2.label}',o=1)
    print(np.mean(A1_scores),np.std(A1_scores),sem(A1_scores),o=1)
    print(np.mean(A2_scores),np.std(A2_scores),sem(A2_scores),o=1)
    print(np.sum(np.sort(B.history[:,:-1],axis=0),axis=1)[-1],'best move each time',o=1)

def some_old_graph_stuff():
    p.imshow(B.history,aspect='auto',cmap='gist_gray',interpolation='none')
    p.scatter(range(num_cycles),best_move(B),s=100,label='best move')
    p.scatter(range(num_cycles),A1.action_history,s=66,label=A1.label)
    p.scatter(range(num_cycles),A2.action_history,s=33,label=A2.label)
    # p.colorbar()
    p.legend()
    p.grid(False)
    # p.axes().set_aspect('equal')
    # p.axes().set_aspect(10)
    p.show()


    p.imshow(np.sort(B.history[:,:-1],axis=0),aspect='auto',cmap='gist_gray',interpolation='none')
    ix = np.argsort(B.history[:,:-1],axis=0)
    i1,i2 = np.where((ix - A1.action_history)==0)
    # print(ix)
    # print(A1.action_history)
    p.scatter(0,0)
    # print(i1)
    # print(i2)
    p.scatter(i2,i1,s=66,label=A1.label)
    i1,i2 = np.where((ix - A2.action_history)==0)
    p.scatter(i2,i1,s=33,label=A2.label)
    # p.scatter(range(num_cycles),A2.action_history,s=75,label=A2.label)
    # p.colorbar()
    p.legend()
    p.grid(False)
    p.show()

def current_scores(A1,A2,A1_scores,A2_scores):
    print('********************************************')
    print('********************************************')
    print('********************************************')
    print(f'{A1.score} - {A1.__class__.__name__}',o=1)
    print(f'{A2.score} - {A2.__class__.__name__}',o=1)
    print(np.mean(A1_scores),np.std(A1_scores),sem(A1_scores),o=1)
    print(np.mean(A2_scores),np.std(A2_scores),sem(A2_scores),o=1)
    A1_wins = np.array(A1_scores)>np.array(A2_scores)
    A2_wins = np.array(A1_scores)<np.array(A2_scores)
    draws = np.array(A1_scores)==np.array(A2_scores)
    print(np.sum(A2_wins)/(np.sum([A1_wins,A2_wins])),f'% wins for {A2.__class__.__name__}')
    print(f'{A1.__class__.__name__} wins: {np.sum(A1_wins)}, {A2.__class__.__name__} wins: {np.sum(A2_wins)}, draws: {np.sum(draws)}')
    # print(f'{np.mean(A2_wins[-100:])} mean % wins for {A2.__class__.__name__} from last 100 games')
    print(f'{np.sum(A2_wins[-100:])/(np.sum(A1_wins[-100:])+np.sum(A2_wins[-100:]))} mean % wins for {A2.__class__.__name__} from last 100 games')
    # print('********************************************')
    print('********************************************')
    print('********************************************')

@timing.timing()
def benchmark(Bandits,benchmark_models,last_vs_all=False,num_ban=100,num_cycles=2000,N_trials=1000,parallel=False,verbose=False,seed=0):
    np.seterr(all='ignore')
    scores = {}
    if last_vs_all:
        comb = [(m,benchmark_models[-1]) for m in benchmark_models[:-1]]
    else:
        comb = list(it.combinations(benchmark_models,2))
    for ii,(A1,A2) in enumerate(comb):
        print(f'{ii} of {len(comb)}')
        models = Parallel(n_jobs=1)(delayed(main)(num_cycles,
                        Bandits(num_ban,num_cycles,rand_state=i+seed+1,verbose=False,parallel=parallel),
                        A1(num_ban,rand_state=i+seed+2),
                        A2(num_ban,rand_state=i+seed+3),
                        verbose=False) for i in range(N_trials))
        A1_scores = []
        A2_scores = []
        for A1,A2 in models:
            A1_scores.append(A1.score)
            A2_scores.append(A2.score)
        A1_wins = np.sum(np.array(A1_scores)>np.array(A2_scores))
        A2_wins = np.sum(np.array(A1_scores)<np.array(A2_scores))
        draws = np.sum(np.array(A1_scores)==np.array(A2_scores))
        A1_mean,A1_std,A1_se = np.round(np.mean(A1_scores)),np.round(np.std(A1_scores)),np.round(sem(A1_scores))
        A2_mean,A2_std,A2_se = np.round(np.mean(A2_scores)),np.round(np.std(A2_scores)),np.round(sem(A2_scores))
        scores[(A1.__class__.__name__,A2.__class__.__name__)] = A1_wins-A2_wins,A1_wins,A2_wins,draws,A1_mean,A1_std,A1_se,A2_mean,A2_std,A2_se
    a = np.zeros((len(benchmark_models),len(benchmark_models)))
    keys = {m(num_ban=100).__class__.__name__:i for i,m in enumerate(benchmark_models)}
    # print(keys)
    print(scores)
    for k,v in scores.items():
        a[keys[k[0]],keys[k[1]]] = v[0]
        a[keys[k[1]],keys[k[0]]] = -v[0]
    p.imshow(a,cmap='PiYG')
    p.xticks(range(len(benchmark_models)),keys,rotation=90)
    p.yticks(range(len(benchmark_models)),keys)
    p.colorbar()
    p.grid(False)
    p.title(f'Number of trials: {N_trials}\n(Read as: left model beats bottom model if its green)')
    p.show()


@timing.timing()
def trial(Bandits,trial_models,num_ban=100,num_cycles=2000,N_trials=1000,parallel=False,verbose=False,verbose_summary=True,seed=0,plot=False,A1_params={},A2_params={},no_opponent=False):
    np.seterr(all='ignore')
    A1_scores = []
    A2_scores = []
    if parallel:
        models = Parallel(n_jobs=-1)(delayed(main)(num_cycles,
                        Bandits(num_ban,num_cycles,rand_state=i+seed+1,verbose=False,parallel=parallel),
                        trial_models[0](num_ban,rand_state=i+seed+2),
                        trial_models[1](num_ban,rand_state=i+seed+3),
                        verbose=False) for i in range(N_trials))
        for A1,A2 in models:
            A1_scores.append(A1.score)
            A2_scores.append(A2.score)
        current_scores(A1,A2,A1_scores,A2_scores)
    else:
        for i in range(N_trials):
            if verbose_summary:
                print(f'trial num: {i}')
            B = Bandits(num_ban,num_cycles,rand_state=i+seed+1,verbose=verbose_summary)
            A1 = trial_models[0](num_ban,rand_state=i+seed+2,params=A1_params)
            A2 = trial_models[1](num_ban,rand_state=i+seed+3,params=A2_params)
            if no_opponent:
                B2 = Bandits(num_ban,num_cycles,rand_state=i+seed+1,verbose=True)
                # same seed as A1
                A2 = trial_models[1](num_ban,rand_state=i+seed+2,params=A2_params)
                main_no(num_cycles,B,B2,A1,A2,verbose)
            else:
                main(num_cycles,B,A1,A2,verbose)
            A1_scores.append(A1.score)
            A2_scores.append(A2.score)
            if verbose_summary:
                current_scores(A1,A2,A1_scores,A2_scores)


            if plot:
                # print(A1.reward_history)
                p.plot(np.cumsum(A1.reward_history),lw=2,alpha=0.5,label=A1.__class__.__name__)
                p.plot(np.cumsum(A2.reward_history),lw=2,alpha=0.5,label=A2.__class__.__name__)
                p.scatter(range(num_cycles),np.cumsum(A1.reward_history),c=A1.action_history,label=A1.__class__.__name__,cmap='hsv',zorder=10,s=100)
                p.scatter(range(num_cycles),np.cumsum(A2.reward_history),c=A2.action_history,label=A2.__class__.__name__,cmap='hsv',zorder=10,s=50)
                A1_text = pd.DataFrame(np.array([np.cumsum(A1.reward_history),A1.action_history]).T,columns=['reward','bandit'])
                A2_text = pd.DataFrame(np.array([np.cumsum(A2.reward_history),A2.action_history]).T,columns=['reward','bandit'])
                # print(A1_text)
                # print(A1_text.shift())
                # print(A1_text.shift()['bandit'] != A1_text['bandit'])
                # print
                # raise
                A1_text = A1_text.loc[A1_text.shift()['bandit'] != A1_text['bandit']]
                A2_text = A2_text.loc[A2_text.shift()['bandit'] != A2_text['bandit']]
                for iii,r in A1_text.iterrows():
                    p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
                for iii,r in A2_text.iterrows():
                    p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
                print(list(A1_text['bandit'].values))
                print()
                print()
                print(list(A2_text['bandit'].values))
                p.legend()
                p.title(f'{A1.__class__.__name__}:{A1.score},{A2.__class__.__name__}:{A2.score}')
                p.show()
                # summary_junk(B,A1,A2)
    current_scores(A1,A2,A1_scores,A2_scores)


@timing.timing()
def trial_RL(Bandits,trial_models,num_ban=100,num_cycles=2000,N_trials=1000,parallel=False,verbose=False,verbose_summary=True,seed=0,plot=False,A1_params={},A2_params={},no_opponent=False):
    np.seterr(all='ignore')
    A1_scores = []
    A2_scores = []
    # A2 = trial_models[1](num_ban,rand_state=seed+3,params=A2_params)
    A2 = trial_models[1](num_ban,rand_state=i+seed+2,params=A2_params)
    for i in range(N_trials):
        if verbose_summary and i%verbose_summary==0:
            print(f'trial num: {i}')
        B = Bandits(num_ban,num_cycles,rand_state=seed+1,verbose=i%verbose_summary==0)
        # B = Bandits(num_ban,num_cycles,rand_state=seed+i,verbose=i%verbose_summary==0)
        A1 = trial_models[0](num_ban,rand_state=i+seed+2,params=A1_params)
        if no_opponent:
            B2 = Bandits(num_ban,num_cycles,rand_state=i+seed+1,verbose=True)
            # same seed as A1
            main_no(num_cycles,B,B2,A1,A2,verbose)
        else:
            main(num_cycles,B,A1,A2,verbose)
        A1_scores.append(A1.score)
        A2_scores.append(A2.score)
        # if verbose_summary and i%100==0:
        if verbose_summary and i%verbose_summary==0:
            current_scores(A1,A2,A1_scores,A2_scores)

        # if plot:
        #     # print(A1.reward_history)
        #     p.plot(np.cumsum(A1.reward_history),lw=2,alpha=0.5,label=A1.__class__.__name__)
        #     p.plot(np.cumsum(A2.reward_history),lw=2,alpha=0.5,label=A2.__class__.__name__)
        #     p.scatter(range(num_cycles),np.cumsum(A1.reward_history),c=A1.action_history,label=A1.__class__.__name__,cmap='hsv',zorder=10,s=100)
        #     p.scatter(range(num_cycles),np.cumsum(A2.reward_history),c=A2.action_history,label=A2.__class__.__name__,cmap='hsv',zorder=10,s=50)
        #     A1_text = pd.DataFrame(np.array([np.cumsum(A1.reward_history),A1.action_history]).T,columns=['reward','bandit'])
        #     A2_text = pd.DataFrame(np.array([np.cumsum(A2.reward_history),A2.action_history]).T,columns=['reward','bandit'])
        #     # print(A1_text)
        #     # print(A1_text.shift())
        #     # print(A1_text.shift()['bandit'] != A1_text['bandit'])
        #     # print
        #     # raise
        #     A1_text = A1_text.loc[A1_text.shift()['bandit'] != A1_text['bandit']]
        #     A2_text = A2_text.loc[A2_text.shift()['bandit'] != A2_text['bandit']]
        #     for iii,r in A1_text.iterrows():
        #         p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
        #     for iii,r in A2_text.iterrows():
        #         p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
        #     print(list(A1_text['bandit'].values))
        #     print()
        #     print()
        #     print(list(A2_text['bandit'].values))
        #     p.legend()
        #     p.title(f'{A1.__class__.__name__}:{A1.score},{A2.__class__.__name__}:{A2.score}')
        #     p.show()
        #     # summary_junk(B,A1,A2)

        A2.reset(A1)

    if plot:
        # print(A1.reward_history)
        # p.plot(np.cumsum(np.array(A2_scores) - np.array(A1_scores)),lw=2,alpha=0.5)
        p.plot(np.mean((np.array(A2_scores) - np.array(A1_scores)).reshape(100,-1),axis=0),lw=2,alpha=0.5)
        # p.plot(np.mean((np.array(A2_scores) > np.array(A1_scores)).reshape(int(len(A1_scores)/10),-1),axis=0),lw=2,alpha=0.5)
        # p.plot(np.convolve(np.cumsum(np.array(A2_scores) - np.array(A1_scores)), np.ones(100)/100, mode='valid'),lw=2,alpha=0.5)
        # np.convolve(x, np.ones(N)/N, mode='valid')

        # p.plot(np.cumsum(A2_scores),lw=2,alpha=0.5,label=A2.__class__.__name__)
        # p.scatter(range(num_cycles),np.cumsum(A1.reward_history),c=A1.action_history,label=A1.__class__.__name__,cmap='hsv',zorder=10,s=100)
        # p.scatter(range(num_cycles),np.cumsum(A2.reward_history),c=A2.action_history,label=A2.__class__.__name__,cmap='hsv',zorder=10,s=50)
        # A1_text = pd.DataFrame(np.array([np.cumsum(A1.reward_history),A1.action_history]).T,columns=['reward','bandit'])
        # A2_text = pd.DataFrame(np.array([np.cumsum(A2.reward_history),A2.action_history]).T,columns=['reward','bandit'])
        # print(A1_text)
        # print(A1_text.shift())
        # print(A1_text.shift()['bandit'] != A1_text['bandit'])
        # print
        # # raise
        # A1_text = A1_text.loc[A1_text.shift()['bandit'] != A1_text['bandit']]
        # A2_text = A2_text.loc[A2_text.shift()['bandit'] != A2_text['bandit']]
        # for iii,r in A1_text.iterrows():
        #     p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
        # for iii,r in A2_text.iterrows():
        #     p.text(iii,r['reward'],r['bandit'],rotation=0,zorder=1000)
        # print(list(A1_text['bandit'].values))
        # print()
        # print()
        # print(list(A2_text['bandit'].values))
        # p.legend()
        # p.title(f'{A1.__class__.__name__}:{A1.score},{A2.__class__.__name__}:{A2.score}')
        p.show()

    current_scores(A1,A2,A1_scores,A2_scores)
    print(len(A2.Q))
    for k,v in A2.Q.items():
        print(k,v,A2.Q_count[k])
        time.sleep(0.01)

def search(Bandits,trial_models,num_ban=100,num_cycles=2000,N_trials=1000,seed=0,params={},num_params=20,no_opponent=False):
    filename = str(datetime.datetime.now()).replace(' ','_').split('.')[0]
    np.seterr(all='ignore')
    rrr = np.random.RandomState(seed)
    scores = []
    sc_m = []
    datas = []
    for ii in range(num_params):
        print(f'params num: {ii}')
        A1_scores = []
        A2_scores = []
        param_choice = {k:rrr.choice(v) for k,v in params.items()}
        for i in range(N_trials):
            # print(f'trial num: {i}')
            B = Bandits(num_ban,num_cycles,rand_state=i+seed+1+ii*999,verbose=False)
            A1 = trial_models[0](num_ban,rand_state=i+seed+2+ii*999)
            A2 = trial_models[1](num_ban,rand_state=i+seed+3+ii*999,params=param_choice)

            if no_opponent:
                B2 = Bandits(num_ban,num_cycles,rand_state=i+seed+1+ii*999,verbose=False)
                # same seed as A1
                A2 = trial_models[1](num_ban,rand_state=i+seed+2+ii*999,params=param_choice)
                main_no(num_cycles,B,B2,A1,A2,verbose=False)
            else:
                main(num_cycles,B,A1,A2,verbose=False)
            A1_scores.append(A1.score)
            A2_scores.append(A2.score)
        A1_wins = np.sum(np.array(A1_scores)>np.array(A2_scores))
        A2_wins = np.sum(np.array(A1_scores)<np.array(A2_scores))
        param_choice['score'] = A2_wins - A1_wins
        # scores.append(A2_wins - A1_wins)
        # sc_m.append(param_choice)
        # ix = np.argsort(scores)[::-1]
        print(f'A1: {A1.__class__.__name__}, A2: {A2.__class__.__name__}')
        # for iii in ix:
        #     print(scores[iii],sc_m[iii])
        datas.append(param_choice)
        data = pd.DataFrame(datas)
        # data = data[data.columns[::-1]]
        data = data[['score'] + list(params.keys())]
        print(data.sort_values('score',ascending=False))
        print('saving...')
        data.to_csv(f'data/{filename}.csv')
        print('saving complete')


def heatmaps(filename):
    data = pd.read_csv(filename)

    plots = list(it.combinations(list(data)[2:],2))[::-1]
    # print(list(plots))
    for i,j in plots:
        p.scatter(data[i],data[j],c=data['score'],s=200,alpha=0.75,cmap='plasma',vmin=0,vmax=100)
        # p.imshow(data[i,j],c=data['score'],s=200,alpha=0.75,cmap='plasma')
        p.xlabel(i)
        p.ylabel(j)
        p.colorbar()
        p.show()

# @timing.timing_p
def main(num_cycles,B,A1,A2,verbose=False):
    for time_step in range(num_cycles):
        B.time_step = A1.time_step = A2.time_step = time_step
        A1.choice = A1.take_action(A2,B)
        A2.choice = A2.take_action(A1,B)


        A1.reward = B.get_reward(A1)
        A2.reward = B.get_reward(A2)
        if verbose:
            print(f'time_step: {time_step}, A1: {A1.choice},{int(A1.reward)} A2: {A2.choice},{int(A2.reward)}     A1 SCORE: {A1.score + A1.reward}    A2 SCORE: {A2.score + A2.reward}',o=1)

        A1.update(A2,B)
        A2.update(A1,B)

        B.update(A1,A2)
    return A1,A2

def main_no(num_cycles,B1,B2,A1,A2,verbose=False):
    for time_step in range(num_cycles):
        B1.time_step = B2.time_step = A1.time_step = A2.time_step = time_step
        A1.choice = A1.take_action(A2,B1)
        A2.choice = A2.take_action(A1,B2)


        A1.reward = B1.get_reward(A1)
        A2.reward = B2.get_reward(A2)
        if verbose:
            print(f'time_step: {time_step}, A1: {A1.choice},{int(A1.reward)} A2: {A2.choice},{int(A2.reward)}     A1 SCORE: {A1.score + A1.reward}    A2 SCORE: {A2.score + A2.reward}',o=1)

        A1.update(A2,B1)
        A2.update(A1,B2)

        B1.update(A1)
        B2.update(A2)
    return A1,A2

