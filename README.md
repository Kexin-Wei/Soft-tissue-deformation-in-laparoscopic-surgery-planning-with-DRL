# spiningup
# Required
- Spinningup: [Installation Guide](https://spinningup.openai.com/en/latest/user/installation.html)
- PyRep: [Installation Guide](https://github.com/stepjam/PyRep)
  - CoppeliaSim: download the lastest version from [official website](https://www.coppeliarobotics.com/downloads)
  - if CoppeliaSim crash after install ubuntu package, show error [#98](https://github.com/stepjam/PyRep/issues/98)
    ```
    ImportError: libcoppeliaSim.so.1: cannot open shared object file: No such file or directory
    ```
    may consider use python 3.7, not 3.6 as spinningup required.
    
# Reward shaping:
- test t8 on test-i7(ddpg) and drl-ubuntu(ppo)
    - ddpg(test-i7):
      - s0: not learn yet, best dist 0.03-0.04
      - s1: learning from 20 ep, best dist 0.03-0.04
    - ppo(drl-ubuntu)
      - s0:learning from 0 ep, best dist 0.03
      - s1:learning from 0 ep, best diet 0.06
- test t9 on test-amd
    - Wait until 200 epochs
    - ddpg: 3000/50
      - s0: not learn yet
    - ppo
      - s0: learning from 0 ep

- test new ddpg parameter on test i7 
  - Wait test t8 finished at least 200 epochs
  - 3000/80, pi_lr = 3e-4


