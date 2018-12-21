#!/usr/bin/env python
# encoding: utf-8

import paramiko
import logging
import os

from arc.settings import servers, check_status_command, submit_command, submit_filename, delete_command
from arc.exceptions import InputError

##################################################################


class SSH_Client(object):
    """
    This is a class for communicating with servers
    """
    def __init__(self, server=''):
        """
        `server` is a key (string) for the servers dictionary
        """
        if server == '':
            raise ValueError('A server name must be specified')
        if server not in servers.keys():
            raise ValueError('Server name invalid. Currently defined servers are: {0}'.format(servers.keys()))
        self.server = server
        self.address = servers[server]['adddress']
        self.un = servers[server]['un']
        self.key = servers[server]['key']
        # logger = paramiko.util.logging.getLogger()
        # logger.setLevel(50)
        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def send_command_to_server(self, command, remote_path=''):
        """
        Send commands to server. `command` is an array of string commands to send.
        If remote_path is not an empty string, the command will be executed in the directory path it points to.
        Returns lists of stdin, stdout, stderr corresponding to the commands sent.
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        try:
            ssh.connect(hostname=self.address, username=self.un)
        except:
            return '', 'paramiko failed to connect'
        if remote_path != '':
            # execute command in remote_path directory. Since each `.exec_command()` is a single session,
            # this must to be added to all commands.
            command = 'cd {0}'.format(remote_path) + '; ' + command
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout = stdout.readlines()
        stderr = stderr.readlines()
        ssh.close()
        return stdout, stderr

    def upload_file(self, remote_file_path, local_file_path='', file_string=''):
        """
        Upload `local_file_path` or the contents of `file_string` to `remote_file_path`.
        Either `file_string` or `local_file_path` must be given.
        """
        if local_file_path and not os.path.isfile(local_file_path):
            raise InputError('Cannot upload a non-existing file.'
                             ' Check why file in path {0} is missing.'.format(local_file_path))
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        ssh.connect(hostname=self.address, username=self.un)
        sftp = ssh.open_sftp()
        with sftp.open(remote_file_path, "w") as f_remote:
            if file_string:
                f_remote.write(file_string)
            elif local_file_path:
                # with open(local_file_path, 'r') as f_local:
                #     f_remote.write(f_local.readlines())
                sftp.put(localpath=local_file_path, remotepath=remote_file_path)
            else:
                raise ValueError('Could not upload file to server. Either `file_string` or `local_file_path`'
                                 ' must be specified')
        sftp.close()
        ssh.close()

    def download_file(self, remote_file_path, local_file_path):
        """
        Download a file from `remote_file_path` to `local_file_path`.
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        ssh.connect(hostname=self.address, username=self.un)
        sftp = ssh.open_sftp()
        sftp.get(remotepath=remote_file_path, localpath=local_file_path)
        sftp.close()
        ssh.close()

    def read_remote_file(self, remote_path, filename):
        """
        Read a remote file. `remote_path` is the remote path (required), a `filename` is also required.
        Returns the file's content.
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        ssh.connect(hostname=self.address, username=self.un)
        sftp = ssh.open_sftp()

        full_path = os.path.join(remote_path, filename)

        with sftp.open(full_path, "r") as f_remote:
            content = f_remote.readlines()
        sftp.close()
        ssh.close()
        return content

    def check_job_status(self, job_id):
        """
        Possible statuses: `before_submission`, `running`, `errored on node xx`, `done`
        Status line formats:
        pharos: '540420 0.45326 xq1340b    user_name       r     10/26/2018 11:08:30 long1@node18.cluster'
        rmg: '14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06'
        """
        cmd = check_status_command[servers[self.server]['cluster_soft']] + ' -u ' + servers[self.server]['un']
        stdout, stderr = self.send_command_to_server(cmd)
        for status_line in stdout:
            if str(job_id) in status_line:
                break
        else:
            return 'done'
        status = status_line.split()[4]
        if status.lower() == 'r':
            return 'running'
        else:
            if self.server == 'pharos':
                return 'errored on node ' + status_line.split()[-1].split('@')[1].split('.')[0][-2:]
            elif self.server == 'rmg':
                return 'errored on node ' + status_line.split()[-1][-2:]
            else:
                raise ValueError('Unknown server {0}'.format(self.server))

    def delete_job(self, job_id):
        """
        Deletes a running job
        """
        cmd = delete_command[servers[self.server]['cluster_soft']] + ' ' + str(job_id)
        self.send_command_to_server(cmd)

    def check_running_jobs_ids(self):
        """
        Return a list of ``int`` representing job IDs of all jobs submited by the user on a server
        """
        running_jobs_ids = list()
        cmd = check_status_command[servers[self.server]['cluster_soft']] + ' -u ' + servers[self.server]['un']
        stdout, stderr = self.send_command_to_server(cmd)
        for i, status_line in enumerate(stdout):
            if (self.server == 'rmg' and i > 0) or (self.server == 'pharos' and i > 1):
                running_jobs_ids.append(int(status_line.split()[0]))
        return running_jobs_ids

    def submit_job(self, remote_path):
        job_status = ''
        job_id = 0
        cmd = submit_command[servers[self.server]['cluster_soft']] + ' ' + submit_filename[servers[self.server]['cluster_soft']]
        stdout, stderr = self.send_command_to_server(cmd, remote_path)
        if 'submitted' in stdout[0].lower():
            job_status = 'running'
            if self.server == 'pharos':
                job_id = int(stdout[0].split()[2])
            elif self.server == 'rmg':
                job_id = int(stdout[0].split()[3])
        if len(stderr) > 0:
            job_status = 'errored'
        return job_status, job_id

# TODO: troubleshoot for job stuck on pharos in bad node. Also, if rmgs says 'priority', change node until 8
# TODO: delete scratch files of a failed job: ssh nodeXX; rm scratch/dhdhdhd/job_number