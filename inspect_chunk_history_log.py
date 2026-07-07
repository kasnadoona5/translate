import sys
import paramiko

def run_ssh_command(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    return exit_status, out, err

def main():
    host = "138.124.26.69"
    user = "root"
    secret = "F4F7cmJW9uiA"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, username=user, password=secret, timeout=30)
        encoding = sys.stdout.encoding or 'ascii'
        
        print("=== CHUNK 0 LOG LOGS ===")
        # Get 1500 lines of compose logs and filter for Chunk 0 (case insensitive)
        status, out, err = run_ssh_command(ssh, "cd /opt/translate && docker-compose logs --tail=1500")
        
        lines = out.splitlines()
        found = False
        for line in lines:
            if "chunk 0" in line.lower() or "chunk_index: 0" in line.lower() or "failed to translate" in line.lower():
                print(line.encode(encoding, errors='replace').decode(encoding))
                found = True
        if not found:
            print("No Chunk 0 log entries found in the last 1500 lines.")
                
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
