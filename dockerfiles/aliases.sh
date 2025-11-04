# personalised shortcuts – sourced for every login shell
alias rc='source ~/.bashrc'
alias rce='nano ~/.bashrc'
alias erc='nano ~/.bashrc'

# micromamba drop-ins
alias mamba='micromamba'
alias conda='micromamba'
alias deact='micromamba deactivate'

# env activators
alias rmge='micromamba activate rmg_env'
alias arce='micromamba activate arc_env'

# code roots (set once at image build)
export rmgpy_path=/home/mambauser/Code/RMG-Py
export rmgdb_path=/home/mambauser/Code/RMG-database
export arc_path=/home/mambauser/Code/ARC

alias rmgcode='cd "$rmgpy_path"'
alias dbcode='cd "$rmgdb_path"'
alias arcode='cd "$arc_path"'

# job wrappers
alias rmg='python "$rmgpy_path/rmg.py" input.py  > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)'
alias arkane='python "$rmgpy_path/Arkane.py" input.py  > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)'
alias arc='python "$arc_path/ARC.py" input.yml  > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)'
alias arcrestart='python "$arc_path/ARC.py" restart.yml > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)'
