#!/usr/bin/env python3
# encoding: utf-8

"""
This module executes a simulation workflow where systems are drawn from
a database (in this case, hdf5 data store) and results are stored in
the same database.

The workflow is designed to be safe for highly parallel execution
with zero job-specific details needing to be supplied by the user
at the time of job submission to the HPC queue.

This is the main workflow that is executed in parallel fashion job arrays.
Concurrency concerns, such as race conditions when reading/writing the
pandas HDF5 file, are handled via POSIX file locks.
The module contains a series of operations that are conditional on the
"status" of an entry in the pandas dataframe, and all the work done here
would be problem-specific.

Code adapted from a script generously provided by bbarnes, ARL
"""

# import os
# import sys
# import fcntl
# import subprocess
# import pandas as pd
# from rdkit import Chem
#
# # dummy external script names below
# from another_script import make_conformer, get_atom_positions, get_atom_types
# from a_project_specific_script import do_qm
#
#
# # if doing continuous opt, each pip should run a single species
# # if problematic: then pipe should set cpu = cpu - 1
#
#
# class Lock:
#     """
#     This is a very convenient way to handle locking the hdf5 file
#     so that different parallel jobs do not access it simultaneously.
#     see: https://web.archive.org/web/20180313170424/http://blog.vmfarms.com/2011/03/cross-process-locking-and.html
#
#     There are other ways to solve this problem for example,
#     real databases handle locking/concurrency
#     see: https://docs.python.org/3/library/sqlite3.html, https://www.sqlite.org/lockingv3.html
#     or a python context manager.
#
#     Args:
#         file_path (str): The path to the file to control.
#     """
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         # this will create it if it does not exist already
#         self.handle = open(file_path, 'w')
#
#     # bitwise OR fcntl.LOCK_NB if you need a non-blocking lock
#     def acquire(self):
#         """get the lock on a file"""
#         fcntl.flock(self.handle, fcntl.LOCK_EX)
#
#     def release(self):
#         """release the lock on a file"""
#         fcntl.flock(self.handle, fcntl.LOCK_UN)
#
#     def __del__(self):
#         self.handle.close()
#
#
# def process_next():
#     """
#     Process the next thread.
#     """
#     lock_file_path = '/path/df.lock'
#     lock = Lock(file_path=lock_file_path)
#
#     # the general "try, except, finally" pattern below with hdf5 file lock
#     # will be used multiple times and could be consolidated into its own
#     # function, or a class (as per SafeHDFStore)
#     # but I had not yet done that in this development work
#     next_item = None
#     try:
#         lock.acquire()
#         df = pd.read_hdf('dfstore.h5', 'dfstore')
#         # find the next unattempted item to process
#         items = df.index[df['status'] == 0]
#         if len(items) > 0:
#             ns_loc = items[0]
#             # get the representation of your molecular system that is being
#             # processed; in my case, I simply get a SMILES string
#             next_item = df.iloc[ns_loc]['items']
#             # note that you're working on it and write to disk
#             print(f'working on {ns_loc} {next_smiles}')
#             # see status codes in makestore.py
#             df.loc[df['smiles'] == next_smiles, 'status'] = 1
#             df.to_hdf('dfstore.h5', 'dfstore', format='table')
#         else:
#             print('nothing to do')
#     except:
#         raise Exception('lock section error')
#     finally:
#         lock.release()
#
#     if next_item is not None:
#         # I use SMILES, but you may have some other representation for your
#         # systems to be evaluated in subsequent calculations
#         m = next_smiles
#         # make a rdkit molecule from the SMILES
#         m = Chem.MolFromSmiles(str(m))
#         # make_conformer generates and optimizes 1000 conformers using rdkit
#         # this is necessary to get xyz coordinates for further calculations
#         # for this demo I assume you have your own way of getting xyz coords
#         m = make_conformer(m)
#         # this gets the geometry of the lowest energy conformer
#         my_conf = m.GetConformers()[0]
#         # the next few lines get traditional xyz-style info
#         pos = get_atom_positions(my_conf)
#         elements = get_atom_types(my_conf)
#         pos = [[-float(coor[k]) for k in range(3)] for coor in pos]
#         # now that we have 3D info, we do any needed QM calculations
#         # this is just a placeholder function (do_qm) for the demo script
#         # your own project-specific heavy calculations will be done in it
#         # values returned are a boolean (qm_worked) and various floating points
#         # do_qm is likely to be executing parallel work (like a MPI job)
#         # that could be done via os.system() / subprocess call, or a pythonic
#         # multiprocessing scheme such as mpi4py, concurrent.futures, dask, etc.
#         qm_worked, result_1, result_2 = do_qm(next_smiles, elements, pos)
#
#         # do_qm should be doing work in a directory specific to the molecule(s)
#         # being processed, therefore it will probably contain code
#         # similar in function to the following
#         smile_string = next_smiles
#         dname2 = smile_string.replace('(', 'Q').split()
#         dname = dname2[0].replace(')', 'Z').split()
#         os.chdir(dname[0])
#         # the above few lines create a string that is corresponds to a SMILES
#         # but replaces characters that are problematic for easily making dirs
#         # we assume do_qm has already made the relevant directory
#         # and stored any of its calculation-specific files there
#         # but the code above shows a scheme to not have all files in one dir
#         # it can be easily modified to have a user-defined subdirectory tree
#
#         if qm_worked:
#             try:
#                 lock.acquire()
#                 df = pd.read_hdf('dfstore.h5', 'dfstore')
#                 df.loc[df['smiles'] == next_smiles, 'status'] = 2
#                 df.loc[df['smiles'] == next_smiles, 'result_1'] = result_1
#                 df.loc[df['smiles'] == next_smiles, 'result_2'] = result_2
#             except:
#                 raise Exception('error on lock after qm success')
#             finally:
#                 df.to_hdf('dfstore.h5', 'dfstore', format='table')
#                 print('qm success, store updated')
#                 lock.release()
#         else:
#             try:
#                 lock.acquire()
#                 df = pd.read_hdf('dfstore.h5', 'dfstore')
#                 df.loc[df['smiles'] == next_smiles, 'status'] = 3
#             except:
#                 raise Exception('error on lock after qm failure')
#             finally:
#                 df.to_hdf('dfstore.h5', 'dfstore', format='table')
#                 print('qm failed')
#                 lock.release()
#                 sys.exit()
#
#         # we know qm worked if we did not sys.exit()
#         # optionally process another calculation
#         # let's call it calc2.  it could also be QM, or anything, really.
#         # for development purposes you may want to make calc2_section = False
#         # for testing other parts of the code without executing this section
#         calc2_section = True
#         if calc2_section:
#
#             try:
#                 lock.acquire()
#                 df = pd.read_hdf('dfstore.h5', 'dfstore')
#                 # note that you're working on it and write to disk
#                 df.loc[df['smiles'] == next_smiles, 'status'] = 4
#                 df.to_hdf('dfstore.h5', 'dfstore', format='table')
#             except:
#                 raise Exception('lock section error')
#             finally:
#                 lock.release()
#
#             # the next few lines show a way to run and parse non-python calcs
#             # this is an alternative to doing work via imported python function
#             calc2_cmd = "source calc2.sh; /path/calc2_binary calc2.in >& calc2.out"
#             # this is doing heavy lifting and the command line
#             # could be written to do MPI parallel work
#             subprocess.run(calc2_cmd, shell=True)
#             # this is an alternative to using grep to search output text files
#             for line in open('calc2.out', 'r'):
#                 if "Result 3" in line:
#                     res3_str = line
#                 if "Result 4" in line:
#                     res4_str = line
#
#             try:
#                 # this is a way to extract items by location from text lines
#                 result_3 = float(res3_str.split()[3])
#                 result_4 = float(res4_str.split()[4])
#             except:
#                 try:
#                     lock.acquire()
#                     df = pd.read_hdf('dfstore.h5', 'dfstore')
#                     df.loc[df['smiles'] == next_smiles, 'status'] = 6
#                 except:
#                     raise Exception('error on lock after calc2 failure')
#                 finally:
#                     df.to_hdf('dfstore.h5', 'dfstore', format='table')
#                     print('calc2 failed')
#                     lock.release()
#                     sys.exit()
#
#             # the conditional below checks to see if the results are of the
#             # types expected before writing them to the database
#             # alternatively, you could further parse calc2.out
#             # to get a result like qm_worked, but it would be calc2_worked
#             if(result_3 > 0 and isinstance(result_4, float)):
#                 try:
#                     lock.acquire()
#                     df = pd.read_hdf('dfstore.h5', 'dfstore')
#                     df.loc[df['smiles'] == next_smiles, 'status'] = 5
#                     df.loc[df['smiles'] == next_smiles, 'result_3'] = result_3
#                     df.loc[df['smiles'] == next_smiles, 'result_4'] = result_4
#                 except:
#                     raise Exception('error on lock after calc2 success')
#                 finally:
#                     df.to_hdf('dfstore.h5', 'dfstore', format='table')
#                     print('calc2 success, store updated')
#                     print("smiles, result_3, result_4")
#                     print(next_smiles, result_3, result_4)
#                     lock.release()
#             else:
#                 try:
#                     lock.acquire()
#                     df = pd.read_hdf('dfstore.h5', 'dfstore')
#                     df.loc[df['smiles'] == next_smiles, 'status'] = 6
#                 except:
#                     raise Exception('error on lock after calc2 failure')
#                 finally:
#                     df.to_hdf('dfstore.h5', 'dfstore', format='table')
#                     print('calc2 failed')
#                     lock.release()
#                     sys.exit()
#
#             # immediately above, process_next() ends.  calc2 has finished.
#             # naturally, as many calculations as you need may be strung
#             # together in a complicated conditional workflow
#             # you may want to add calc3, calc4a, calc4b, etc.
#
# # this demo assumes that process_next() will start with an unattempted
# # molecule/system and execute the entire workflow in one HPC job
# # however, it is possible to add further conditionals (and functions) in order
# # to restart / continue jobs based upon the status code of the molecular system
# # in the pandas hdf5 file.  therefore you could execute a long workflow
# # automatically, even if it was interrupted due to queue time limits
#
# # the function process_next() is called to do the work
# # after the class Lock and process_next() definitions
# # this may safely be called in independent, duplicate parallel jobs
# process_next()
