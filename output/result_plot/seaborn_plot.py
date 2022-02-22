from posixpath import join
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_csv(dir_name,case_name):
    file_name = []
    # find .csv file
    base_name = os.path.join(os.path.dirname( __file__ ),"..")
    join_name = os.path.join(base_name, dir_name, case_name)

    for file in os.listdir(join_name):
        if file.endswith(".csv"):
            file_name.append( os.path.join(join_name,file) )
    return file_name

def get_min_data(file_name,set_min=None):
    # numpy
    min_len = 1e6
    data = []
    for i in range(len(file_name)):
        temp = pd.read_csv(file_name[i]).to_numpy()
        if min_len>temp.shape[0]: min_len = temp.shape[0] 
        data.append(temp)

    if set_min is not None and set_min<min_len: min_len = set_min
    new_data = np.zeros((min_len,len(file_name)))
    for i in range(len(data)):
        new_data[:,i]=data[i][:min_len,-1]
    return new_data

def save_sns_plot(sns_plot, fig_title):
    sns_plot.set(title=fig_title)
    sns_plot.figure.savefig( os.path.join(os.path.dirname( __file__ ),f"{fig_title}.png"))



def liver_p_to_p():
    dir_name = "liver_p_to_p"
    case_name = "2021-12-18_ppo"
    fig_title = "Point-to-Point without Collision"
    file_name = get_csv(dir_name,case_name)

    new_data = get_min_data(file_name,set_min=500)
 
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10,5))
    new_data = pd.DataFrame(data=new_data,index=np.arange(new_data.shape[0]),columns=[f"data" for i in range(len(file_name))])
    sns_plot = sns.lineplot(data=new_data,legend=False)
    save_sns_plot(sns_plot,fig_title)
    print("done")

def pure_p_to_p():
    dir_name = 'pure_p_to_p'
    case_names = ['2021-12-18_ddpg','2021-12-18_ppo']
    fig_title = "Point-to-Point without Soft Tissue"
    file_name = []
    for case_name in case_names:
        file_name += get_csv(dir_name,case_name)
    print(file_name)
    del file_name[-1]

    data = get_min_data(file_name,set_min=200)

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10,5))
    # ax = plt.gca()
    # new_data = []
    # sigma = 20

    # for i in range(len(file_name)):
    #     d = data[:,i]
        
    #     # Smoothing
    #     smooth_data = gaussian_filter1d(d, sigma=sigma)

    #     # Error formating
    #     upper_err = smooth_data + sigma
    #     lower_err = smooth_data - sigma
    #     ax.plot(np.arange(d.shape[0]),d,'--')
    #     ax.plot(np.arange(d.shape[0]),smooth_data)
    #     ax.fill_between(np.arange(d.shape[0]),upper_err,lower_err)
    #     #df = pd.DataFrame(data=np.c_[smooth_data,upper_err,lower_err],columns=["smooth","upper_err","lower_err"])
    #     #new_data.append(df)
    # plt.show(block=False)
    new_data = pd.DataFrame(data=data,index=np.arange(data.shape[0]),columns=["DDPG","PPO"])
    sns_plot = sns.lineplot(data=new_data)
    save_sns_plot(sns_plot,fig_title)
    print("done")
    
def train_ddpg():
    dir_name = 'ddpg_test_amd'
    case_names = ['2021-12-19_10-11-01-ddpg_s0_t5', '2021-12-19_01-15-44-ddpg_s0_should_t5']
    fig_title = "Training Curve of DDPG"

    file_name = []
    for case_name in case_names:
        file_name += get_csv(dir_name,case_name)
    print(file_name)

    new_data = get_min_data(file_name,set_min=250)

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10,5))
    new_data = pd.DataFrame(data=new_data,index=np.arange(new_data.shape[0]))
    sns_plot = sns.lineplot(data=new_data,legend=False)
    save_sns_plot(sns_plot,fig_title)
    print("done")

def train_ppo():
    dir_name = 'ppo_test_amd'
    case_names = ['2021-12-19_15-00-57-ppo_s0_outliner','2021-12-21_11-09-31-ppo_s0_outliner']
    fig_title = "Training Curve of PPO"
    file_name = []
    for case_name in case_names:
        file_name += get_csv(dir_name,case_name)
    print(file_name)

    new_data = get_min_data(file_name,set_min=250)
 
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10,5))
    new_data = pd.DataFrame(data=new_data,index=np.arange(new_data.shape[0]),columns=[f"data" for i in range(len(file_name))])
    sns_plot = sns.lineplot(data=new_data,legend=False)
    save_sns_plot(sns_plot,fig_title)
    print("done")

def train_ppo_ddpg_sep():
    fig_title = "PPO_DDPG_SEP"

    dir_name = 'ppo_test_amd'
    case_names = ['2021-12-19_15-00-57-ppo_s0_outliner','2021-12-21_11-09-31-ppo_s0_outliner']
    ppo_file_name = []
    for case_name in case_names:
        ppo_file_name += get_csv(dir_name,case_name)
    
    dir_name = 'ddpg_test_amd'
    case_names = ['2021-12-19_10-11-01-ddpg_s0_t5', '2021-12-19_01-15-44-ddpg_s0_should_t5']
    ddpg_file_name = []
    for case_name in case_names:
        ddpg_file_name += get_csv(dir_name,case_name)

    ppo_data  = get_min_data(ppo_file_name,set_min=250)
    ddpg_data = get_min_data(ddpg_file_name,set_min=250)
    data = np.c_[ppo_data,ddpg_data]
    new_data = pd.DataFrame(data=data,index=np.arange(data.shape[0]), columns=['PPO1','PPO2','DDPG1','DDPG2'])
    #new_data['index'] = new_data.index
      
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(7,5))
    sns_plot=sns.lineplot(data=new_data)
    sns_plot.set_xlabel("Epochs")                        
    sns_plot.set_ylabel("Reward")
    sns_plot.set(title="Training Curve for PPO and DDPG")
    sns_plot.figure.savefig( os.path.join(os.path.dirname( __file__ ),f"{fig_title}.png"))
    print("done")

def train_ppo_ddpg_group():
    fig_title = "PPO_DDPG_GROUP"

    dir_name = 'ppo_test_amd'
    case_names = ['2021-12-19_15-00-57-ppo_s0_outliner','2021-12-21_11-09-31-ppo_s0_outliner']
    ppo_file_name = []
    for case_name in case_names:
        ppo_file_name += get_csv(dir_name,case_name)
    
    dir_name = 'ddpg_test_amd'
    case_names = ['2021-12-19_10-11-01-ddpg_s0_t5', '2021-12-19_01-15-44-ddpg_s0_should_t5']
    ddpg_file_name = []
    for case_name in case_names:
        ddpg_file_name += get_csv(dir_name,case_name)

    ppo_data  = get_min_data(ppo_file_name,set_min=250)
    ddpg_data = get_min_data(ddpg_file_name,set_min=250)
    ppo_df = pd.DataFrame(data=ppo_data)
    ddpg_df = pd.DataFrame(data=ddpg_data)
    ppo_df['type']='PPO'
    ddpg_df['type']='DDPG'
    data = pd.concat([ppo_df,ddpg_df])
    draw_df = data.reset_index().melt(id_vars=['index', 'type'], var_name='col')
    
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(7,5))
    sns_plot=sns.lineplot(  x='index', 
                            y='value',
                            hue='type', 
                            #palette=['b','r'],
                            data=draw_df)
    sns_plot.set_xlabel("Epochs")                        
    sns_plot.set_ylabel("Reward")
    sns_plot.set(title="Training Curve for PPO and DDPG")
    sns_plot.figure.savefig( os.path.join(os.path.dirname( __file__ ),f"{fig_title}.png"))
    print("done")

def train_ppo_ddpg_side():
    fig_title = "PPO_DDPG_SIDE"

    dir_name = 'ppo_test_amd'
    case_names = ['2021-12-19_15-00-57-ppo_s0_outliner','2021-12-21_11-09-31-ppo_s0_outliner']
    ppo_file_name = []
    for case_name in case_names:
        ppo_file_name += get_csv(dir_name,case_name)
    
    dir_name = 'ddpg_test_amd'
    case_names = ['2021-12-19_10-11-01-ddpg_s0_t5', '2021-12-19_01-15-44-ddpg_s0_should_t5']
    ddpg_file_name = []
    for case_name in case_names:
        ddpg_file_name += get_csv(dir_name,case_name)
    
    ppo_data  = get_min_data(ppo_file_name,set_min=250)
    ddpg_data = get_min_data(ddpg_file_name,set_min=250)

    ppo_df  = pd.DataFrame(data=ppo_data,index=np.arange(ppo_data.shape[0]),columns=[f"data" for i in range(len(ppo_file_name))])
    ddpg_df = pd.DataFrame(data=ddpg_data,index=np.arange(ddpg_data.shape[0]))
    
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 2,sharey=True,figsize=(7,5))
    fig.suptitle("Training Curve for PPO and DDPG")
    sns.lineplot(data=ppo_df,legend=False,ax=axes[0])
    sns.lineplot(data=ddpg_df,legend=False,ax=axes[1])
    axes[0].set_title("PPO")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Epochs")
    axes[1].set_title("DDPG")
    axes[1].set_xlabel("Epochs")
    fig.savefig(os.path.join(os.path.dirname( __file__ ),f"{fig_title}.png"))
    print("done")

if __name__ == "__main__":
    #liver_p_to_p()
    #pure_p_to_p()
    #train_ddpg()
    #train_ppo()
    train_ppo_ddpg_sep()
    train_ppo_ddpg_group()
    train_ppo_ddpg_side()
   
