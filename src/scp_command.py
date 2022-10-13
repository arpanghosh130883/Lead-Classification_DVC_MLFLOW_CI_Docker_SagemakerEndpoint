import paramiko
from scp import SCPClient

def createSSHClient(server, user):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient(ec2-13-41-157-208, ubuntu)
scp = SCPClient(ssh.get_transport())