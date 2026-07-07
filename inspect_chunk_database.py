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
        
        print("=== DETAILED CHUNK ROWS IN DB ===")
        db_script = (
            "docker exec -i translate_tarjomeh_1 python -c \"\n"
            "import sqlite3\n"
            "conn = sqlite3.connect('jobs/jobs.db')\n"
            "conn.row_factory = sqlite3.Row\n"
            "c = conn.cursor()\n"
            "rows = c.execute('SELECT * FROM chunks ORDER BY chunk_index ASC LIMIT 4').fetchall()\n"
            "for r in rows:\n"
            "    d = dict(r)\n"
            "    print(' ') \n"
            "    print('--- ChunkIndex:', d.get('chunk_index'), 'Status:', d.get('status'), '---')\n"
            "    for key in ['id', 'job_id', 'status', 'text', 'translation', 'score']:\n"
            "        val = d.get(key)\n"
            "        if key in ('text', 'translation'):\n"
            "            print('  ', key, ':', val[:120].strip().replace('\\n', ' ') + '...' if val else 'None')\n"
            "        else:\n"
            "            print('  ', key, ':', val)\n"
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
