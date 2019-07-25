#!/usr/bin/env python
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from arc.job.ssh import delete_all_arc_jobs
from arc.job.local import delete_all_local_arc_jobs
from arc.settings import servers

################################################################################

"""
Caution!!
Using this function might results in deletion of jobs running on all servers ARC has access to (including 'local').
Use this only if you are certain in what you're doing to avoid deleting valuable jobs and information loss.
"""


def main():
    """
    Delete all ARC jobs from all servers.
    """
    server_list = [server for server in servers.keys() if server != 'local']
    delete_all_arc_jobs(server_list=server_list)
    if 'local' in servers:
        delete_all_local_arc_jobs()


################################################################################

if __name__ == '__main__':
    main()
