#!/usr/bin/env python3
# encoding: utf-8

"""
ARC - Automatic Rate Calculator

Caution!!
Using this module might results in deletion of jobs running on all servers ARC has access to (including 'local').
Use this only if you are certain in what you're doing to avoid deleting valuable jobs and information loss.
"""

import argparse
import csv
import os

from arc.common import ARC_PATH
from arc.exceptions import InputError
from arc.imports import local_arc_path, settings
from arc.job.local import delete_all_local_arc_jobs
from arc.job.ssh import delete_all_arc_jobs


servers = settings['servers']


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.
    """

    parser = argparse.ArgumentParser(description='ARC delete util')

    parser.add_argument('-p', '--project', type=str, nargs=1, default='',
                        metavar='Project', help='the project for which all jobs wil be deleted')
    parser.add_argument('-j', '--job', type=str, nargs=1, default='',
                        metavar='Job', help='the job the belongs to a project for which all jobs wil be deleted')
    parser.add_argument('-s', '--server', type=str, nargs=1, default='',
                        metavar='Server', help='the server name from which to delete jobs')
    parser.add_argument('-a', '--all', action='store_true', help='delete all ARC jobs')

    args = parser.parse_args(command_line_args)
    if args.job:
        if args.job[0][0] == 'a':
            args.job = args.job[0][1:]
        else:
            args.job = args.job[0]
    if args.project:
        args.project = args.project[0]

    return args


def main():
    """
    Delete ARC jobs according to the command line arguments specifications.
    """

    args = parse_command_line_arguments()

    if not args.all and not args.project and not args.job and not args.server:
        raise InputError("Either a project (e,g,, '-p project_name'), a job (e.g., '-j a4563'), "
                         "or a server (e,g,, '-s server_name'), or ALL (i.e., '-a')")

    server_list = args.server if args.server else [server for server in servers.keys()]

    local_arc_path_ = local_arc_path if os.path.isdir(local_arc_path) else ARC_PATH
    csv_path = os.path.join(local_arc_path_, 'initiated_jobs.csv')

    project, jobs = None, list()
    if args.project:
        project = args.project
    elif args.job:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, dialect='excel')
            for row in reader:
                if args.job == row[0]:
                    project = row[1]

    if project is not None:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, dialect='excel')
            for row in reader:
                if row[1] == project:
                    jobs.append(row[8])

    if args.all:
        jobs = None

    for server in server_list:
        if server != 'local':
            delete_all_arc_jobs(server_list=server_list, jobs=jobs)
        else:
            delete_all_local_arc_jobs(jobs=jobs)


if __name__ == '__main__':
    main()
