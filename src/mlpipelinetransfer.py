# import scp

# files = ['stage_01_dataprepare.py', 'stage_02_modeltraining.py', 'stage_03_modelevaluation.py', 'stage_04_testing_script.py', 'stage_05_black_formatting.py', 'stage_06_vulture_stdcoding.py']

# scp -i ./mlpipeline.pem -r ./src/* ubuntu@ec2-13-41-157-208.eu-west-2.compute.amazonaws.com:~/.



# client = scp.Client(host=host, user=user, keyfile=keyfile)
# or
# client = scp.SCPClient(host='ec2-13-41-157-208', user='ubuntu')
# client.use_system_keys()
# or
# client = scp.Client(host=host, user=user, password=password)

# and then
# client.transfer('/etc/local/filename', '/etc/remote/filename')




# for file in files:
#     client.transfer('./file', 'ubuntu@ec2-13-41-157-208.eu-west-2.compute.amazonaws.com:~/.')

    # scp -i ./mlpipeline.pem -r ./src/file ubuntu@ec2-13-41-157-208.eu-west-2.compute.amazonaws.com:~/.

from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(hostname='ec2-13-42-47-209.eu-west-2.compute.amazonaws.com', 
            port = '22',
            username='ubuntu',
            # password='password',
            pkey='load_key_if_relevant')


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

scp.put('./src/testing.py', 'ubuntu@ec2-13-41-157-208.eu-west-2.compute.amazonaws.com:~/.')
# scp.get('file_path_on_remote_machine', 'file_path_on_local_machine')

scp.close()


"""

from paramiko import SSHClient
from scp import SCPClient

ssh_ob = SSHClient()
ssh_ob.load_system_host_keys()
ssh_ob.connect('ubuntu@ec2-13-41-157-208.eu-west-2.compute.amazonaws.com')
scp = SCPClient(ssh_ob.get_transport())

scp.put('src/testing.py')
# scp.get('sampletxt2.txt')
# scp.put('sample', recursive=True, remote_path='/home/sample_user')

scp.close()
"""