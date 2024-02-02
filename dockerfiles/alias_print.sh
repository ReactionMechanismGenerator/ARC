#!/bin/bash

# Source .bashrc to load aliases
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Print the aliases
echo "Alias List with Descriptions:"
echo ""
echo "1. rmge ='micromamba activate rmg_env'"
echo "   - Activates the 'rmg_env' environment using Conda."
echo ""
echo "2. arce ='micromamba activate arc_env'"
echo "   - Activates the 'arc_env' environment using Conda."
echo ""
echo "3. rmg ='python-jl /home/rmguser/Code/RMG-Py/rmg.py input.py'"
echo "   - Runs the RMG program with 'input.py' using Python in the Julia environment."
echo ""
echo "4. deact ='micromamba deactivate'"
echo "   - Deactivates the current Micromamba environment."
echo ""
echo "5. rmgcode='cd /home/rmguser/Code/RMG-Py/'"
echo "   - Changes the current directory to the RMG-Py code directory."
echo ""
echo "6. rmgdb ='cd /home/rmguser/Code/RMG-database/"
echo "   - Changes the current directory to the RMG database directory."
echo ""
echo "7. arcode ='cd /home/rmguser/Code/ARC'"
echo "   - Changes the current directory to the ARC code directory."
echo ""

# Execute the command specified as CMD in Dockerfile, or the command passed to docker run
exec "$@"
