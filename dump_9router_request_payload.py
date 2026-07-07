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
        
        print("=== DUMPING REQUEST PAYLOAD FOR CHUNK 0 ===")
        db_script = (
            "python3 -c \"\n"
            "import sqlite3, json\n"
            "conn = sqlite3.connect('/root/.9router/db/data.sqlite')\n"
            "conn.row_factory = sqlite3.Row\n"
            "c = conn.cursor()\n"
            "row = c.execute('SELECT data FROM requestDetails WHERE id = \\'1783424972967-qzxd6ecso\\'').fetchone()\n"
            "if row:\n"
            "    data_obj = json.loads(row['data'])\n"
            "    req = data_obj.get('request', {})\n"
            "    print(json.dumps(req, indent=2))\n"
            "else:\n"
            "    print('Request ID not found.')\n"
            "\""
        )
        status, out, err = run_ssh_command(ssh, db_script)
        if out.strip():
            print(out.encode(encoding, errors='replace').decode(encoding))
        if err.strip():
            print("[STDERR]", err.encode(encoding, errors='replace').decode(encoding))
        
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
