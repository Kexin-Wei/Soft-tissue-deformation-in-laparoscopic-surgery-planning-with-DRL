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
      - s0: not learn yet
      - s1: learning from 20 ep
    - ppo(drl-ubuntu)
      - s0:learning from 0 ep
      - s1:learning from 0 ep
  - test t9 on test-amd
    - ddpg: 3000/50
      - s0
    - ppo
      - s0

