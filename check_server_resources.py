import paramiko

def check_gpu():
    hostname = "209.38.249.154"
    username = "root"
    password = r"06050022003dD@d"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname=hostname, username=username, password=password, timeout=15)
        stdin, stdout, stderr = client.exec_command("nvidia-smi")
        print("NVIDIA-SMI Output:")
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        stdin, stdout, stderr = client.exec_command("free -h")
        print("Memory Output:")
        print(stdout.read().decode())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_gpu()
