import scipy.io 
import numpy as np
import matplotlib.pyplot as plp
from matplotlib.lines import Line2D 


#import data
mat = scipy.io.loadmat('sample_eeg.mat')


## Loading data from file
training = mat['training']
signal = mat['signal']
num_chans = mat['num_chans']
srate = mat['srate']
events = mat['events']
eventinds = mat['eventinds']
username = mat['username']

#set up conds 
conds =['Calibration',training[0]]
num_conds = len(conds)
calidone = np.array([u'calibdone'], dtype='<U9')

##set up for single calidone event, change to list if multiple
for i in range(0, events.size):
    if np.array_equal(events[0,i], calidone):
        bindex = eventinds[0,i]
               
cond_inds = np.array([[0, bindex], [bindex, signal.size]])

# for each condition calculate moving spectral power
# we split data into moving 1 sec epochs (or trials) every 250ms

# set FFT parameters
T = 1 # epoch length in seconds 
N = T*srate # number of samples per epoch
N = int(N)
df = 1/T  #frequency resolution

asp_t_l = []
for i in range(0,len(conds)):
    asp_t_l.append([])

for cond in range(0,len(conds)):
    tmp = np.array([])
    x = np.array([signal[(cond_inds[cond,0]):(cond_inds[cond,1])]]) # signal for this condition
    sz = x.size # length of signal for this condition
    x_epoched_list =  [] # will hold array of singal split into array of moving overlapping epochs
    ep = 0
    x= np.squeeze(x)
    
    for t in range(0,sz,0.25*srate):
        if (t+N) <= sz: # make sure that we have an epoch of length N
            ep = ep + 1 # signal for this epoch only
            tmp = x[t + np.arange(N)]
            tmp = tmp-np.mean(tmp)
            x_epoched_list.append([tmp])
    x_epoched = np.array(x_epoched_list)
    x_epoched = np.squeeze(x_epoched)
            
    #end of t
    
    trialcount = len(x_epoched) # number of epochs/trials 
    

    # here we store asp in {cond}(freq trial) format
    
    #asp_t{cond} = zeros(ceil(N/2),trialcount); 
    for trial in range(0,trialcount):
        # apply FFT to this trial
        b = np.squeeze(x_epoched[trial,]) #(time chan)
        fftb = abs(np.fft.fft(b,N)) # (freq chan)
        fftb = np.power(fftb,2)
        asp_ep = fftb[0:int(np.ceil(N/2))] #(freq chan)
        asp_t_l[cond].append(asp_ep) # {cond}(freq trial)
        

asp_t = np.squeeze((np.array(asp_t_l)))

# plot asp across trials for specific freq bands
wind = 100 # smoothing window across trials, for display purposes
#define frequency bands: 1-3Hz, 4-8Hz, ...

bands = np.array([[1,3],[4,8],[8,12],[13,18],[19,30],[31,45]])
leg = ['delta','theta','alpha','beta13-18','beta19-30','gamma']
cols = ['b','g','r','c','k','y']

#figure('name',['username : ' username]); hold on;

lastind = 0 # keep track of trial indices, as we move through conditions

#for loop below
tobe_sqz = []
lower = int(round(T*1))
higher = int(round(T*50)) +1 
fig = plp.figure()
ax = fig.add_subplot(1, 1, 1)
ax.figsize=(20,10)

for cond in np.arange(num_conds):
    # for relative spec power calculate total power 1-50 Hz
    asp_t[cond,] = np.swapaxes(asp_t[cond,],0,1)
    tobe_sqz = np.sum(asp_t[cond,][lower:higher], axis=0)
    totpow = np.squeeze(tobe_sqz) # (trial)
    
    # now calculate relative power for each freq band
    for b in range(bands.shape[0]):
        tmp = np.squeeze(np.sum(asp_t[cond,][int(round(T*bands[b,0])):int(round(T*bands[b,1]))+1],axis=0))
        tmp = np.divide(tmp,totpow)
        tmp = tmp.transpose()
        ax.plot(np.arange(lastind,lastind+(tmp).size),tmp, cols[b]+'-', lw=.3)


    # annotate condition name
    plp.annotate(str(conds[cond]), xy=(lastind + tmp.size/2,0.6), xytext=(lastind + tmp.size/2,.75),
                rotation=90)    
     
     #if this is last condition, display legend
    if cond != num_conds -1 : 
        lastind = lastind + tmp.size;
        ax.plot([lastind, lastind],[0,0.7 ], c='k', lw= 2)        
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.legend(leg)    
ax.set_ylabel('abs spectral power')  
ax.set_xlabel('Trials')
fig = plp.gcf()
fig.canvas.set_window_title('Username: '+ username[0])
fig.savefig("python.pdf")
plp.show()


